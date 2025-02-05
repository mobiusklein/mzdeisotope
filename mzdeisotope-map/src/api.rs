use std::marker::PhantomData;
use std::mem;

use mzdeisotope::charge::ChargeRange;
use mzdeisotope::isotopic_model::CachingIsotopicModel;
use mzdeisotope::scorer::{IsotopicFitFilter, IsotopicPatternScorer};
use mzpeaks::feature::Feature;
use mzpeaks::{coordinate::MZ, feature_map::FeatureMap};
use mzpeaks::{prelude::*, Mass};

use crate::DeconvolutionError;
use crate::{processor::FeatureProcessor, FeatureSearchParams, solution::DeconvolvedSolutionFeature};
use mzdeisotope::IsotopicModelLike;

#[derive(Debug, Clone)]
pub struct FeatureDeconvolutionEngine<
    'lifespan,
    T: Clone + Default,
    C: FeatureLike<MZ, T> + Clone + Default,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
> {
    /// The set of parameters to use for `isotopic_model` when generating an isotopic pattern for a given m/z
    isotopic_params: FeatureSearchParams,
    /// The model to generate isotpoic patterns from, or a collection there-of. If more than one model is
    /// provided, a slightly different algorithm will be used.
    isotopic_model: Option<IsotopicModelLike<'lifespan>>,
    /// The strategy for scoring isotopic pattern fits
    scorer: Option<S>,
    /// The strategy for filtering out isotopic pattern fits that are too poor to consider
    fit_filter: Option<F>,
    peak_type: PhantomData<C>,
    time_dim: PhantomData<T>,
}

impl<
        'lifespan,
        T: Clone + Default,
        C: FeatureLike<MZ, T> + Clone + Default,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > FeatureDeconvolutionEngine<'lifespan, T, C, S, F>
{
    pub fn new<I: Into<IsotopicModelLike<'lifespan>>>(
        isotopic_params: FeatureSearchParams,
        isotopic_model: I,
        scorer: S,
        fit_filter: F,
    ) -> Self {
        Self {
            isotopic_params,
            isotopic_model: Some(isotopic_model.into()),
            scorer: Some(scorer),
            fit_filter: Some(fit_filter),
            peak_type: PhantomData,
            time_dim: PhantomData,
        }
    }


    /// Pre-calculcate and cache all isotopic patterns between `min_mz` and `max_mz` for
    /// charge states between `min_charge` and `max_charge`.
    ///
    /// If this method is not used, experimental peaks will be used to seed the isotopic pattern
    /// caches which may lead to slightly different solutions depending upon the order in which
    /// peak lists are processed.
    ///
    /// # See also
    /// [`IsotopicPatternGenerator::populate_cache`](crate::isotopic_model::IsotopicPatternGenerator::populate_cache)
    pub fn populate_isotopic_model_cache(
        &mut self,
        min_mz: f64,
        max_mz: f64,
        min_charge: i32,
        max_charge: i32,
    ) {
        if let Some(cache) = self.isotopic_model.as_mut() {
            match cache {
                IsotopicModelLike::SingleModel(cache) => {
                    cache.populate_cache_params(
                        min_mz,
                        max_mz,
                        min_charge,
                        max_charge,
                        self.isotopic_params.as_isotopic_params(),
                    );
                }
                IsotopicModelLike::MultipleModels(caches) => {
                    for cache in caches {
                        cache.populate_cache_params(
                            min_mz,
                            max_mz,
                            min_charge,
                            max_charge,
                            self.isotopic_params.as_isotopic_params(),
                        );
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn deconvolute_features(
        &mut self,
        features: FeatureMap<MZ, T, C>,
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        minimum_size: usize,
        maximum_time_gap: f64,
        minimum_intensity: f32,
        max_missed_peaks: usize,
    ) -> Result<FeatureMap<Mass, T, DeconvolvedSolutionFeature<T>>, DeconvolutionError> {
        let output = match mem::take(&mut self.isotopic_model).unwrap() {
            IsotopicModelLike::SingleModel(model) => {
                let mut deconvoluter =
                    FeatureProcessor::<T, CachingIsotopicModel<'lifespan>, S, F>::new(
                        features.into_iter().map(|f| {
                            let f: Feature<MZ, T> = f.iter().collect();
                            f
                        }).collect(),
                        model,
                        mem::take(&mut self.scorer).unwrap(),
                        mem::take(&mut self.fit_filter).unwrap(),
                        minimum_size,
                        maximum_time_gap,
                        minimum_intensity,
                        true
                    );

                let mut params = self.isotopic_params;
                params.max_missed_peaks = max_missed_peaks;

                let output = deconvoluter.deconvolve(
                    error_tolerance,
                    charge_range,
                    1,
                    0,
                    &self.isotopic_params,
                    1e-3,
                    10,
                );

                self.isotopic_model = Some(deconvoluter.isotopic_model.into());
                self.scorer = Some(deconvoluter.scorer);
                self.fit_filter = Some(deconvoluter.fit_filter);
                output
            }
            _ => {
                todo!("not yet implemented")
            }
        };
        output
    }
}

#[allow(clippy::too_many_arguments)]
pub fn deconvolute_features<'a, T: Clone + Default, C: FeatureLike<MZ, T> + Clone + Default, I: Into<IsotopicModelLike<'a>>, S: IsotopicPatternScorer, F: IsotopicFitFilter>(
        features: FeatureMap<MZ, T, C>,
        isotopic_params: FeatureSearchParams,
        isotopic_model: I,
        scorer: S,
        fit_filter: F,
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        minimum_size: usize,
        maximum_time_gap: f64,
        minimum_intensity: f32,
        max_missed_peaks: usize,
) -> Result<FeatureMap<Mass, T, DeconvolvedSolutionFeature<T>>, DeconvolutionError> {
    let mut engine = FeatureDeconvolutionEngine::new(isotopic_params, isotopic_model, scorer, fit_filter);

    engine.deconvolute_features(features, error_tolerance, charge_range, minimum_size, maximum_time_gap, minimum_intensity, max_missed_peaks)
}
