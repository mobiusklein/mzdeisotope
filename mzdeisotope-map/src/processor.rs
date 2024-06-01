use chemical_elements::{isotopic_pattern::TheoreticalIsotopicPattern, neutral_mass};

use itertools::Itertools;
use mzpeaks::{
    coordinate::{CoordinateRange, Span1D},
    feature::{ChargedFeature, Feature, TimeInterval},
    feature_map::FeatureMap,
    prelude::*,
    CoordinateLikeMut, Mass, MZ,
};

use mzdeisotope::{
    charge::{ChargeIterator, ChargeRange, ChargeRangeIter},
    isotopic_model::{
        isotopic_shift, IsotopicPatternGenerator, TheoreticalIsotopicDistributionScalingMethod,
        PROTON,
    },
    scorer::{IsotopicFitFilter, IsotopicPatternScorer, ScoreInterpretation, ScoreType},
};
use thiserror::Error;

use crate::{
    dependency_graph::{DependenceCluster, FeatureDependenceGraph, FitRef, SubgraphSolverMethod},
    feature_fit::{FeatureSetFit, MapCoordinate},
    DeconvolvedSolutionFeature, FeatureSetIter,
};

/// An error that might occur during deconvolution
#[derive(Debug, Clone, PartialEq, Error)]
pub enum DeconvolutionError {
    #[error("Failed to resolve a deconvolution solution")]
    FailedToResolveSolution,
    #[error("Failed to resolve a fit reference {0:?}")]
    FailedToResolveFit(FitRef),
}

#[derive(Debug, Clone, Copy)]
pub struct FeatureSearchParams {
    pub truncate_after: f64,
    pub max_missed_peaks: usize,
    pub threshold_scale: f32,
    pub detection_threshold: f32,
}

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

    pub fn conform<C: CentroidLike + Default + CoordinateLikeMut<MZ> + IntensityMeasurementMut>(
        &self,
        experimental: Vec<Option<C>>,
        theoretical: &mut TheoreticalIsotopicPattern,
    ) -> (Vec<C>, usize) {
        let mut cleaned_eid = Vec::with_capacity(experimental.len());
        let mut n_missing: usize = 0;
        let mut total_intensity = 0.0f32;

        for (fpeak, peak) in experimental.into_iter().zip(theoretical.iter()) {
            if fpeak.is_none() {
                let mut tpeak = C::default();
                *tpeak.coordinate_mut() = peak.mz();
                *tpeak.intensity_mut() = 1.0;
                if peak.intensity > self.minimum_theoretical_abundance {
                    n_missing += 1;
                }
                total_intensity += tpeak.intensity();
                cleaned_eid.push(tpeak)
            } else {
                total_intensity += fpeak.as_ref().unwrap().intensity();
                cleaned_eid.push(fpeak.unwrap())
            }
        }
        theoretical.iter_mut().for_each(|p| {
            p.intensity *= total_intensity as f64;
        });
        (cleaned_eid, n_missing)
    }
}

const MAX_COMBINATIONS: usize = 10000 / 2;

#[derive(Debug)]
pub struct FeatureProcessor<
    Y: Clone,
    I: IsotopicPatternGenerator,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
> {
    pub feature_map: FeatureMap<MZ, Y, Feature<MZ, Y>>,
    pub isotopic_model: I,
    pub scorer: S,
    pub fit_filter: F,
    pub scaling_method: TheoreticalIsotopicDistributionScalingMethod,
    pub prefer_multiply_charged: bool,
    pub minimum_size: usize,
    pub maximum_time_gap: f64,
    pub envelope_conformer: EnvelopeConformer,
    pub dependency_graph: FeatureDependenceGraph,
}

impl<Y: Clone, I: IsotopicPatternGenerator, S: IsotopicPatternScorer, F: IsotopicFitFilter>
    FeatureProcessor<Y, I, S, F>
{
    pub fn new(
        feature_map: FeatureMap<MZ, Y, Feature<MZ, Y>>,
        isotopic_model: I,
        scorer: S,
        fit_filter: F,
        scaling_method: TheoreticalIsotopicDistributionScalingMethod,
        minimum_size: usize,
        maximum_time_gap: f64,
        prefer_multiply_charged: bool,
    ) -> Self {
        let dependency_graph = FeatureDependenceGraph::new(scorer.interpretation());
        Self {
            feature_map,
            isotopic_model,
            scorer,
            fit_filter,
            scaling_method,
            envelope_conformer: EnvelopeConformer::new(0.05),
            minimum_size,
            maximum_time_gap,
            prefer_multiply_charged,
            dependency_graph,
        }
    }

    pub fn find_all_features(
        &self,
        mz: f64,
        error_tolerance: Tolerance,
    ) -> Vec<(usize, &Feature<MZ, Y>)> {
        let indices = self.feature_map.all_indices_for(mz, error_tolerance);
        indices
            .into_iter()
            .map(|i| (i, &self.feature_map[i]))
            .collect_vec()
    }

    pub fn find_features(
        &self,
        mz: f64,
        error_tolerance: Tolerance,
        interval: &Option<CoordinateRange<Y>>,
    ) -> Option<Vec<(usize, &Feature<MZ, Y>)>> {
        let f = self.find_all_features(mz, error_tolerance);
        if f.is_empty() {
            return None;
        } else if let Some(interval) = interval {
            let search_width = interval.end.unwrap() - interval.start.unwrap();
            if search_width == 0.0 {
                return None;
            }
            let f: Vec<(usize, &Feature<MZ, Y>)> = f
                .into_iter()
                .filter(|(_, f)| {
                    let t = f.as_range();
                    if t.overlaps(&interval) {
                        ((t.end.unwrap() - t.start.unwrap()) / search_width) >= 0.05
                    } else {
                        false
                    }
                })
                .collect();
            if f.is_empty() {
                None
            } else {
                Some(f)
            }
        } else {
            Some(f.into_iter().collect_vec())
        }
    }

    pub fn match_theoretical_isotopic_distribution(
        &self,
        theoretical_distribution: &TheoreticalIsotopicPattern,
        error_tolerance: Tolerance,
        interval: &Option<CoordinateRange<Y>>,
    ) -> Vec<Option<Vec<(usize, &Feature<MZ, Y>)>>> {
        theoretical_distribution
            .iter()
            .map(|p| self.find_features(p.mz, error_tolerance, interval))
            .collect()
    }

    pub fn collect_all_fits<Z: ChargeIterator>(
        &mut self,
        feature: usize,
        error_tolerance: Tolerance,
        charge_range: Z,
        left_search: i8,
        right_search: i8,
        search_params: &FeatureSearchParams,
    ) -> usize {
        let (mut best_fit_score, is_maximizing) = match self.scorer.interpretation() {
            ScoreInterpretation::HigherIsBetter => (0.0, true),
            ScoreInterpretation::LowerIsBetter => (ScoreType::INFINITY, false),
        };
        let mut best_fit_charge = 0;

        let mut holdout = None;
        let mut counter = 0;

        for charge in charge_range {
            let current_fits = self.fit_theoretical_distribution(
                feature,
                error_tolerance,
                charge,
                left_search,
                right_search,
                search_params,
            );

            let is_multiply_charged = charge.abs() > 1;
            if is_maximizing {
                for fit in current_fits.iter() {
                    if fit.score > best_fit_score {
                        if is_multiply_charged && !fit.has_multiple_real_features() {
                            continue;
                        }
                        best_fit_score = fit.score;
                        best_fit_charge = charge.abs();
                    }
                }
            } else {
                for fit in current_fits.iter() {
                    if fit.score < best_fit_score {
                        if is_multiply_charged && !fit.has_multiple_real_features() {
                            continue;
                        }
                        best_fit_score = fit.score;
                        best_fit_charge = charge.abs();
                    }
                }
            }
            if !is_multiply_charged && self.prefer_multiply_charged {
                holdout = Some(current_fits);
            } else {
                for fit in current_fits {
                    counter += 1;
                    self.dependency_graph.add_fit(fit);
                }
            }
        }
        if !holdout.is_some() && best_fit_charge == 1 {
            for fit in holdout.unwrap() {
                counter += 1;
                self.dependency_graph.add_fit(fit);
            }
        }
        counter
    }

    pub fn fit_theoretical_distribution(
        &mut self,
        feature: usize,
        error_tolerance: Tolerance,
        charge: i32,
        left_search: i8,
        right_search: i8,
        search_params: &FeatureSearchParams,
    ) -> Vec<FeatureSetFit> {
        let base_mz = self.feature_map.get_item(feature).mz();
        let time_range = Some(self.feature_map.get_item(feature).as_range());
        let mut all_fits = Vec::new();
        for offset in -left_search..=right_search {
            let shift = isotopic_shift(charge) * (offset as f64);
            let mz = base_mz + shift;
            all_fits.extend(self.fit_feature_set(
                mz,
                error_tolerance,
                charge,
                search_params,
                &time_range,
            ));
        }
        all_fits
    }

    fn fit_feature_set(
        &mut self,
        mz: f64,
        error_tolerance: Tolerance,
        charge: i32,
        search_params: &FeatureSearchParams,
        feature: &Option<CoordinateRange<Y>>,
    ) -> Vec<FeatureSetFit> {
        let base_tid = self.isotopic_model.isotopic_cluster(
            mz,
            charge,
            PROTON,
            search_params.truncate_after,
            0.05,
        );

        let fits = self.fit_theoretical_distribution_on_features(
            mz,
            error_tolerance,
            charge,
            base_tid,
            search_params.max_missed_peaks,
            search_params.threshold_scale,
            feature,
        );

        fits
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
        let fgi = feature_groups
            .into_iter()
            .map(|g| {
                if g.is_none() {
                    vec![None].into_iter()
                } else {
                    let v: Vec<_> = g.unwrap().into_iter().map(|f| Some(f)).collect();
                    v.into_iter()
                }
            })
            .multi_cartesian_product()
            .enumerate();
        let mut snapped_tid: TheoreticalIsotopicPattern;
        let mut fits = Vec::new();
        for (i, features) in fgi {
            if i > MAX_COMBINATIONS {
                break;
            }

            if features.iter().all(|f| f.is_none()) {
                continue;
            }

            if let Some((_, f)) = features.first().unwrap() {
                snapped_tid = base_tid.clone().shift(f.mz());
            } else {
                snapped_tid = base_tid.clone().shift(mz);
            };

            let mut counter = 0;
            let mut score_max: ScoreType = 0.0;

            let mut score_vec: Vec<ScoreType> = Vec::new();
            let mut time_vec: Vec<f64> = Vec::new();

            let mut features_vec: Vec<_> = Vec::with_capacity(features.len());
            let mut indices_vec: Vec<_> = Vec::with_capacity(features.len());
            for opt in features.into_iter() {
                if let Some((i, f)) = opt {
                    indices_vec.push(Some(i));
                    features_vec.push(Some(f));
                } else {
                    features_vec.push(None);
                    indices_vec.push(None);
                }
            }

            let it = FeatureSetIter::new(&features_vec);
            for (time, eid) in it {
                counter += 1;
                let mut tid = snapped_tid.clone();
                let (cleaned_eid, n_missing) = self.envelope_conformer.conform(eid, &mut tid);
                if n_missing > max_missed_peaks {
                    continue;
                }

                let score = self.scorer.score(&cleaned_eid, &tid);
                // tracing::debug!("{cleaned_eid:?} vs. {tid} => {score:0.2}");
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
            let missing_features: usize = features_vec.iter().map(|f| f.is_none() as usize).sum();

            let (start, end) = features_vec
                .iter()
                .flatten()
                .map(|f| {
                    let pt = f.iter().next().unwrap();
                    let start = MapCoordinate::new(*pt.0, *pt.1);
                    let pt = f.iter().last().unwrap();
                    let end = MapCoordinate::new(*pt.0, *pt.1);
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
            tracing::debug!("Selecting feature set score {final_score} at mass {neutral_mass:0.2} with charge {charge} {start:?}-{end:?}");
            if !self.fit_filter.test_score(final_score) {
                continue;
            }

            fits.push(fit);
        }

        fits
    }

    fn find_thresholded_score(
        &self,
        scores: &[ScoreType],
        maximum_score: ScoreType,
        percentage: ScoreType,
    ) -> ScoreType {
        let threshold = maximum_score * percentage;
        tracing::debug!("Extracting score over {scores:?} {} items with maximum {maximum_score} with threshold {threshold} ({percentage})", scores.len());
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

    fn skip_feature(&self, feature: &Feature<MZ, Y>) -> bool {
        self.minimum_size < feature.len()
    }

    fn explore_local<Z: ChargeIterator>(
        &mut self,
        feature: usize,
        error_tolerance: Tolerance,
        charge_range: Z,
        left_search: i8,
        right_search: i8,
        search_params: &FeatureSearchParams,
    ) -> usize {
        if self.skip_feature(&self.feature_map[feature]) {
            0
        } else {
            self.collect_all_fits(
                feature,
                error_tolerance,
                charge_range,
                left_search,
                right_search,
                search_params,
            )
        }
    }

    pub fn populate_graph(
        &mut self,
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        left_search: i8,
        right_search: i8,
        search_params: &FeatureSearchParams,
    ) -> usize {
        let n = self.feature_map.len();
        if n == 0 {
            return 0;
        }
        (0..n)
            .rev()
            .map(|i| {
                self.explore_local(
                    i,
                    error_tolerance,
                    ChargeRangeIter::from(charge_range),
                    left_search,
                    right_search,
                    search_params,
                )
            })
            .sum()
    }

    fn solve_subgraph_top(
        &mut self,
        cluster: DependenceCluster,
        fits: Vec<(FitRef, FeatureSetFit)>,
        peak_accumulator: &mut Vec<FeatureSetFit>,
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

    fn select_best_disjoint_subgraphs(
        &mut self,
        fit_accumulator: &mut Vec<FeatureSetFit>,
    ) -> Result<(), DeconvolutionError> {
        let solutions = self
            .dependency_graph
            .solutions(SubgraphSolverMethod::Greedy);
        tracing::debug!("{} distinct solution clusters", solutions.len());
        let res: Result<(), DeconvolutionError> =
            solutions.into_iter().try_for_each(|(cluster, fits)| {
                self.solve_subgraph_top(cluster, fits, fit_accumulator)
            });
        res
    }

    pub fn graph_step_deconvolve(
        &mut self,
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        left_search: i8,
        right_search: i8,
        search_params: &FeatureSearchParams,
    ) -> Result<Vec<FeatureSetFit>, DeconvolutionError> {
        let mut fit_accumulator = Vec::new();
        self.populate_graph(
            error_tolerance,
            charge_range,
            left_search,
            right_search,
            search_params,
        );
        tracing::debug!("{} fits in the graph", self.dependency_graph.fit_nodes.len());
        self.select_best_disjoint_subgraphs(&mut fit_accumulator)?;
        Ok(fit_accumulator)
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
            .map(|i| i.map(|i| self.feature_map.get_item(i)))
            .collect();
        let feat_iter = FeatureSetIter::new_with_time_interval(
            &features,
            time_range.start().unwrap(),
            time_range.end().unwrap(),
        );

        let base_tid = &fit.theoretical;
        let charge = fit.charge;
        let abs_charge = charge.abs();

        // Accumulators for the experimental envelope features
        let mut envelope_features = Vec::with_capacity(base_tid.len());

        // Accumulators for the residual intensities for features
        let mut residuals = Vec::with_capacity(base_tid.len());

        // Initialize the accumulators
        for _ in base_tid.iter() {
            envelope_features.push(Feature::empty());
            residuals.push(Vec::new());
        }

        let mut feature = ChargedFeature::empty(charge);

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
            if !(score < 0.0 || score.is_nan()) {
                continue;
            }

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
                        .push_raw(e.mz(), time, intens);

                    // Update the residual for this time point
                    if e.intensity() * 0.7 < t.intensity() {
                        residuals[i].push(1.0);
                    } else {
                        residuals[i].push((e.intensity() - t.intensity()).max(1.0));
                    }
                });
            let neutral_mass = tid[0].neutral_mass();
            feature.push_raw(neutral_mass, time, total_intensity);
        }

        drop(features);

        // Do the subtraction now that the wide reads are done
        for (i, fidx) in fit.features.iter().enumerate() {
            if fidx.is_none() {
                continue;
            }
            let f = &mut self.feature_map[fidx.unwrap()];
            for (time, res_int) in feature.iter_time().zip(residuals[i].iter().copied()) {
                if let Some((_mz_at, _time_at, int_at)) = f.at_time_mut(time) {
                    *int_at = res_int;
                }
            }
        }
        DeconvolvedSolutionFeature::new(feature, fit.score, envelope_features.into_boxed_slice())
    }

    #[allow(unused)]
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
        let ref_tick = before_tic;
        let mut deconvoluted_features = Vec::new();
        let mut converged = false;
        let mut convergence_check = f32::MAX;
        for i in 0..max_iterations {
            tracing::debug!(
                "Starting iteration {i} with remaining TIC {before_tic:0.4e} ({:0.3}%), {} feature fit",
                before_tic / ref_tick * 100.0,
                deconvoluted_features.len()
            );

            let fits = self.graph_step_deconvolve(
                error_tolerance,
                charge_range,
                left_search_limit,
                right_search_limit,
                search_params,
            )?;

            tracing::debug!("Found {} fits", fits.len());

            let minimum_size = self.minimum_size;
            let max_gap_size = self.maximum_time_gap;
            deconvoluted_features.extend(
                fits.iter()
                    .map(|fit| {
                        self.finalize_fit(
                            fit,
                            search_params.detection_threshold,
                            search_params.max_missed_peaks,
                        )
                    })
                    .map(|f| {
                        let (breakpoints, _) = f.iter_time().fold(
                            (Vec::new(), f.start_time().unwrap()),
                            |(mut points, last_time), time| {
                                if time - last_time > max_gap_size {
                                    points.push(time)
                                }
                                (points, time)
                            },
                        );
                        let mut segments = Vec::new();
                        let mut remainder = f;
                        for i in breakpoints {
                            let (before, after) = remainder.split_at(i);
                            segments.push(before);
                            remainder = after;
                        }
                        segments.push(remainder);
                        segments
                    })
                    .flatten(),
            );

            let after_tic = self.feature_map.iter().map(|f| f.total_intensity()).sum();
            convergence_check = (before_tic - after_tic) / after_tic;
            if convergence_check <= convergence {
                tracing::debug!(
                    "Converged at on iteration {i} with remaining TIC {before_tic:0.4e} - {after_tic:0.4e} = {:0.4e} ({convergence_check}), {} peaks fit",
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
        if !converged {
            tracing::debug!(
                "Failed to converge after {max_iterations} iterations with remaining TIC {before_tic:0.4e} ({convergence_check}), {} peaks fit",
                deconvoluted_features.len()
            );
        }

        Ok(FeatureMap::new(deconvoluted_features))
    }
}
