use chemical_elements::{isotopic_pattern::TheoreticalIsotopicPattern, neutral_mass};

use itertools::Itertools;
use mzpeaks::{
    coordinate::CoordinateRange,
    feature::{Feature, TimeInterval},
    feature_map::FeatureMap,
    prelude::*,
    CoordinateLikeMut, MZ,
};

use mzdeisotope::{
    charge::ChargeIterator,
    isotopic_model::{
        isotopic_shift, IsotopicPatternGenerator, TheoreticalIsotopicDistributionScalingMethod,
        PROTON,
    },
    scorer::{IsotopicFitFilter, IsotopicPatternScorer, ScoreInterpretation, ScoreType},
};

use crate::{
    feature_fit::{FeatureSetFit, MapCoordinate},
    FeatureSetIter,
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
        feature: &Feature<MZ, Y>,
        error_tolerance: Tolerance,
        charge_range: Z,
        left_search: usize,
        right_search: usize,
        truncate_after: f64,
        max_missed_peaks: usize,
        threshold_scale: f32,
    ) -> Vec<FeatureSetFit> {
        let mut fits = Vec::new();
        let (mut best_fit_score, is_maximizing) = match self.scorer.interpretation() {
            ScoreInterpretation::HigherIsBetter => (0.0, true),
            ScoreInterpretation::LowerIsBetter => (ScoreType::INFINITY, false),
        };
        let mut best_fit_charge = 0;

        let mut holdout = None;

        for charge in charge_range {
            let current_fits = self.fit_theoretical_distribution(
                feature,
                error_tolerance,
                charge,
                left_search,
                right_search,
                truncate_after,
                max_missed_peaks,
                threshold_scale,
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
                    //TODO: ("add to dependency network");
                    fits.push(fit);
                }
            }
        }
        if !holdout.is_some() && best_fit_charge == 1 {
            for fit in holdout.unwrap() {
                //TODO: ("add to dependency network");
                fits.push(fit);
            }
        }
        fits
    }

    pub fn fit_theoretical_distribution(
        &mut self,
        feature: &Feature<MZ, Y>,
        error_tolerance: Tolerance,
        charge: i32,
        left_search: usize,
        right_search: usize,
        truncate_after: f64,
        max_missed_peaks: usize,
        threshold_scale: f32,
    ) -> Vec<FeatureSetFit> {
        let base_mz = feature.mz();
        let mut all_fits = Vec::new();
        for offset in -(left_search as isize)..=(right_search as isize) {
            let shift = isotopic_shift(charge) * (offset as f64);
            let mz = base_mz + shift;
            all_fits.extend(self.fit_feature_set(
                mz,
                error_tolerance,
                charge,
                truncate_after,
                max_missed_peaks,
                threshold_scale,
                Some(feature),
            ));
        }
        all_fits
    }

    fn fit_feature_set(
        &mut self,
        mz: f64,
        error_tolerance: Tolerance,
        charge: i32,
        truncate_after: f64,
        max_missed_peaks: usize,
        threshold_scale: f32,
        feature: Option<&Feature<MZ, Y>>,
    ) -> Vec<FeatureSetFit> {
        let base_tid =
            self.isotopic_model
                .isotopic_cluster(mz, charge, PROTON, truncate_after, 0.05);

        let fits = self.fit_theoretical_distribution_on_features(
            mz,
            error_tolerance,
            charge,
            base_tid,
            max_missed_peaks,
            threshold_scale,
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
        feature: Option<&Feature<MZ, Y>>,
    ) -> Vec<FeatureSetFit> {
        let feature_groups = self.match_theoretical_isotopic_distribution(
            &base_tid,
            error_tolerance,
            &feature.map(|f| f.as_range()),
        );
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

        let (acc, count) = scores.iter().fold((0.0, 0usize), |(total, count), val| {
            if *val > threshold {
                (total + val, count + 1)
            } else {
                (total, count)
            }
        });
        acc / count as ScoreType
    }
}
