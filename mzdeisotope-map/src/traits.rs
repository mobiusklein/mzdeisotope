use std::collections::HashMap;

use identity_hash::BuildIdentityHasher;
use itertools::Itertools;

use chemical_elements::isotopic_pattern::TheoreticalIsotopicPattern;

use mzdeisotope::{
    charge::{ChargeIterator, ChargeListIter, ChargeRange},
    isotopic_model::{isotopic_shift, IsotopicPatternParams},
    scorer::{ScoreInterpretation, ScoreType},
};
use mzpeaks::{
    coordinate::CoordinateRange,
    feature::TimeInterval,
    prelude::*,
    MZ,
};
use thiserror::Error;
use tracing::debug;

use crate::{
    dependency_graph::{
        DependenceCluster, FeatureDependenceGraph, FitKey, FitRef, SubgraphSolverMethod,
    }, FeatureSetFit,
    fmap::{IndexedFeatureMap, IndexedFeature}
};

pub(crate) type FeatureType<Y> = IndexedFeature<Y>;
pub(crate) type FeatureMapType<Y> = IndexedFeatureMap<Y>;

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
    pub ignore_below: f64,
    pub max_missed_peaks: usize,
    pub threshold_scale: f32,
    pub detection_threshold: f32,
}

impl Default for FeatureSearchParams {
    fn default() -> Self {
        Self {
            truncate_after: 0.95,
            ignore_below: 0.05,
            max_missed_peaks: 1,
            threshold_scale: 0.3,
            detection_threshold: 0.1,
        }
    }
}

impl FeatureSearchParams {
    pub fn new(
        truncate_after: f64,
        ignore_below: f64,
        max_missed_peaks: usize,
        threshold_scale: f32,
        detection_threshold: f32,
    ) -> Self {
        Self {
            truncate_after,
            ignore_below,
            max_missed_peaks,
            threshold_scale,
            detection_threshold,
        }
    }

    pub fn as_isotopic_params(&self) -> IsotopicPatternParams {
        IsotopicPatternParams {
            incremental_truncation: None,
            truncate_after: self.truncate_after,
            ignore_below: self.ignore_below,
            ..Default::default()
        }
    }
}



pub fn quick_charge_feature<C: FeatureLike<MZ, Y>, Y, const N: usize>(
    features: &[C],
    position: usize,
    charge_range: ChargeRange,
) -> ChargeListIter {
    let (min_charge, max_charge) = charge_range;
    let mut charges = [false; N];
    if N > 0 {
        charges[0] = true;
    }
    let feature = &features[position];
    let mut result_size = 0usize;
    let min_intensity = feature.intensity() / 4.0;
    let query_time = feature.as_range();
    for other in features.iter().skip(position + 1).filter(|f| f.as_range().overlaps(&query_time)) {
        if other.intensity() < min_intensity {
            continue;
        }
        let diff = other.mz() - feature.mz();
        if diff > 1.1 {
            break;
        }
        let raw_charge = 1.0 / diff;
        let charge = (raw_charge + 0.5) as i32;
        let remain = raw_charge - raw_charge.floor();
        if 0.2 < remain && remain < 0.8 {
            continue;
        }
        if charge < min_charge || charge > max_charge {
            continue;
        }
        if !charges[(charge - 1) as usize] {
            result_size += 1;
        }
        charges[(charge - 1) as usize] = true;
    }

    let mut result = Vec::with_capacity(result_size);
    charges.iter().enumerate().for_each(|(j, hit)| {
        let z = (j + 1) as i32;
        if *hit && accept_charge(z, &charge_range) {
            result.push(z)
        }
    });
    result.into()
}

#[inline(always)]
fn accept_charge(z: i32, charge_range: &ChargeRange) -> bool {
    let z = z.abs();
    charge_range.0.abs() <= z && z <= charge_range.1.abs()
}

/// An wrapper around [`quick_charge`] which dispatches to an appropriate staticly compiled
/// variant with minimal stack allocation.
pub fn quick_charge_feature_w<C: FeatureLike<MZ, Y>, Y>(
    peaks: &[C],
    position: usize,
    charge_range: ChargeRange,
) -> ChargeListIter {
    macro_rules! match_i {
        ($($i:literal, )*) => {
            match charge_range.1 {
                $($i => quick_charge_feature::<C, Y, $i>(peaks, position, charge_range),)*
                i if i > 16 && i < 33 => quick_charge_feature::<C, Y, 32>(peaks, position, charge_range),
                i if i > 32 && i < 65 => quick_charge_feature::<C, Y, 64>(peaks, position, charge_range),
                i if i > 64 && i < 129 => quick_charge_feature::<C, Y, 128>(peaks, position, charge_range),
                _ => quick_charge_feature::<C, Y, 256>(peaks, position, charge_range),
            }
        };
    }
    match_i!(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,)
}

pub trait FeatureMapMatch<Y> {
    fn feature_map(&self) -> &FeatureMapType<Y>;
    fn feature_map_mut(&mut self) -> &mut FeatureMapType<Y>;

    fn find_all_features(
        &self,
        mz: f64,
        error_tolerance: Tolerance,
    ) -> Vec<(usize, &FeatureType<Y>)> {
        let indices = self.feature_map().all_indices_for(mz, error_tolerance);
        indices
            .into_iter()
            .map(|i| (i, &self.feature_map()[i]))
            .collect_vec()
    }

    fn find_features(
        &self,
        mz: f64,
        error_tolerance: Tolerance,
        interval: &Option<CoordinateRange<Y>>,
    ) -> Option<Vec<(usize, &FeatureType<Y>)>> {
        let f = self.find_all_features(mz, error_tolerance);
        if f.is_empty() {
            None
        } else if let Some(interval) = interval {
            let search_width = interval.end.unwrap() - interval.start.unwrap();
            if search_width == 0.0 {
                return None;
            }
            let f: Vec<(usize, &FeatureType<Y>)> = f
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

    fn match_theoretical_isotopic_distribution(
        &self,
        theoretical_distribution: &TheoreticalIsotopicPattern,
        error_tolerance: Tolerance,
        interval: &Option<CoordinateRange<Y>>,
    ) -> Vec<Option<IndexedIsotopicFitFeatureSet<'_, Y>>> {
        theoretical_distribution
            .iter()
            .map(|p| self.find_features(p.mz, error_tolerance, interval))
            .collect()
    }
}


pub type IndexedIsotopicFitFeatureSet<'a, Y> = Vec<(usize, &'a FeatureType<Y>)>;

pub trait FeatureIsotopicFitter<Y>: FeatureMapMatch<Y> {
    fn fit_theoretical_distribution(
        &mut self,
        feature: usize,
        error_tolerance: Tolerance,
        charge: i32,
        left_search: i8,
        right_search: i8,
        search_params: &FeatureSearchParams,
    ) -> Vec<FeatureSetFit> {
        let (base_mz, time_range) = {
            let feature = self.feature_map().get_item(feature);
            let base_mz = feature.mz();
            let time_range = Some(feature.as_range());
            (base_mz, time_range)
        };
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

    fn make_isotopic_pattern(
        &mut self,
        mz: f64,
        charge: i32,
        search_params: &FeatureSearchParams,
    ) -> TheoreticalIsotopicPattern;

    fn fit_feature_set(
        &mut self,
        mz: f64,
        error_tolerance: Tolerance,
        charge: i32,
        search_params: &FeatureSearchParams,
        feature: &Option<CoordinateRange<Y>>,
    ) -> Vec<FeatureSetFit> {
        let base_tid = self.make_isotopic_pattern(mz, charge, search_params);



        self.fit_theoretical_distribution_on_features(
            mz,
            error_tolerance,
            charge,
            base_tid,
            search_params.max_missed_peaks,
            search_params.threshold_scale,
            feature,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn fit_theoretical_distribution_on_features(
        &self,
        mz: f64,
        error_tolerance: Tolerance,
        charge: i32,
        base_tid: TheoreticalIsotopicPattern,
        max_missed_peaks: usize,
        threshold_scale: f32,
        feature: &Option<CoordinateRange<Y>>,
    ) -> Vec<FeatureSetFit>;
}

pub trait GraphFeatureDeconvolution<Y>: FeatureIsotopicFitter<Y> {
    fn score_interpretation(&self) -> ScoreInterpretation;

    fn add_fit_to_graph(&mut self, fit: FeatureSetFit) {
        self.dependency_graph_mut().add_fit(fit);
    }

    fn prefer_multiply_charged(&self) -> bool;

    fn skip_feature(&self, feature: &FeatureType<Y>) -> bool;

    fn dependency_graph_mut(&mut self) -> &mut FeatureDependenceGraph;

    fn populate_graph(
        &mut self,
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        left_search: i8,
        right_search: i8,
        search_params: &FeatureSearchParams,
    ) -> usize {
        let n = self.feature_map().len();
        if n == 0 {
            return 0;
        }
        (0..n)
            .rev()
            .map(|i| {
                if i % 5000 == 0 {
                    debug!(
                        "Processing feature {}/{n} ({:0.2}%)",
                        (n - i),
                        (n - i) as f32 / n as f32 * 100.0
                    )
                }
                let charge_range_of = quick_charge_feature_w(
                    self.feature_map().as_slice(),
                    i,
                    charge_range
                );
                self.explore_local(
                    i,
                    error_tolerance,
                    charge_range_of,
                    left_search,
                    right_search,
                    search_params,
                )
            })
            .sum()
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
        if self.skip_feature(self.feature_map().get_item(feature)) {
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

    fn collect_all_fits<Z: ChargeIterator>(
        &mut self,
        feature: usize,
        error_tolerance: Tolerance,
        charge_range: Z,
        left_search: i8,
        right_search: i8,
        search_params: &FeatureSearchParams,
    ) -> usize {
        let (mut best_fit_score, is_maximizing) = match self.score_interpretation() {
            ScoreInterpretation::HigherIsBetter => (0.0, true),
            ScoreInterpretation::LowerIsBetter => (ScoreType::INFINITY, false),
        };
        let mut best_fit_charge = 0;
        let prefer_multiply_charged = self.prefer_multiply_charged();
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
            if !is_multiply_charged && prefer_multiply_charged {
                holdout = Some(current_fits);
            } else {
                for fit in current_fits {
                    if is_multiply_charged && !fit.has_multiple_real_features() {
                        continue;
                    }
                    counter += 1;
                    if counter % 100 == 0 && counter > 0 {
                        debug!("Added {counter}th solution for feature {feature} to graph");
                    }
                    self.add_fit_to_graph(fit);
                }
            }
        }
        if holdout.is_some() && best_fit_charge == 1 {
            for fit in holdout.unwrap() {
                counter += 1;
                self.add_fit_to_graph(fit);
            }
        }
        counter
    }

    fn solve_subgraph_top(
        &mut self,
        cluster: DependenceCluster,
        fits: Vec<(FitRef, FeatureSetFit)>,
        fit_accumulator: &mut Vec<FeatureSetFit>,
    ) -> Result<(), DeconvolutionError> {
        let mut fits: HashMap<FitKey, FeatureSetFit, BuildIdentityHasher<FitKey>> =
            fits.into_iter().map(|(k, v)| (k.key, v)).collect();
        for best_fit_key in cluster.iter().take(1) {
            if let Some(fit) = fits.remove(&best_fit_key.key) {
                fit_accumulator.push(fit);
            } else {
                return Err(DeconvolutionError::FailedToResolveFit(*best_fit_key));
            }
        }
        Ok(())
    }

    fn select_best_disjoint_subgraphs(
        &mut self,
        fit_accumulator: &mut Vec<FeatureSetFit>,
    ) -> Result<(), DeconvolutionError> {
        let solutions = self
            .dependency_graph_mut()
            .solutions(SubgraphSolverMethod::Greedy);
        debug!("{} distinct solution clusters", solutions.len());
        let res: Result<(), DeconvolutionError> =
            solutions.into_iter().try_for_each(|(cluster, fits)| {
                self.solve_subgraph_top(cluster, fits, fit_accumulator)
            });
        res
    }

    fn graph_step_deconvolve(
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
        tracing::debug!(
            "{} fits in the graph",
            self.dependency_graph_mut().fit_nodes.len()
        );
        self.select_best_disjoint_subgraphs(&mut fit_accumulator)?;
        Ok(fit_accumulator)
    }
}
