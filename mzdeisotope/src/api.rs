//! * High level APIs for running deconvolution operations
#![allow(clippy::too_many_arguments)]

use std::marker::PhantomData;
use std::mem;

use mzpeaks::{
    peak_set::PeakSetVec, CentroidLike, CentroidPeak, IntensityMeasurementMut, MZPeakSetType,
    MassPeakSetType, Tolerance,
};

use crate::{
    charge::ChargeRange,
    deconv_traits::{
        DeconvolutionError, IsotopicDeconvolutionAlgorithm, IsotopicPatternFitter,
        TargetedDeconvolution,
    },
    deconvoluter::GraphDeconvoluterType,
    isotopic_model::{CachingIsotopicModel, IsotopicPatternParams},
    multi_model_deconvoluters::GraphMultiDeconvoluterType,
    scorer::{IsotopicFitFilter, IsotopicPatternScorer},
    solution::DeconvolvedSolutionPeak,
};

/// An algebraic data structure for abstracting over one or many
/// [`CachingIsotopicModel`].
#[derive(Debug, Clone)]
pub enum IsotopicModelLike<'a> {
    SingleModel(CachingIsotopicModel<'a>),
    MultipleModels(Vec<CachingIsotopicModel<'a>>),
}

impl<'a, I: Into<CachingIsotopicModel<'a>>> From<I> for IsotopicModelLike<'a> {
    fn from(value: I) -> Self {
        Self::SingleModel(value.into())
    }
}

impl<'a, I: Into<CachingIsotopicModel<'a>>> From<Vec<I>> for IsotopicModelLike<'a> {
    fn from(value: Vec<I>) -> Self {
        if value.is_empty() {
            panic!("Attempted to convert an empty collection into an isotopic model parameter")
        } else if value.len() == 1 {
            Self::SingleModel(unsafe { value.into_iter().next().unwrap_unchecked().into() })
        } else {
            Self::MultipleModels(value.into_iter().map(|v| v.into()).collect())
        }
    }
}

impl<'a, I: Into<CachingIsotopicModel<'a>>> FromIterator<I> for IsotopicModelLike<'a> {
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        (iter
            .into_iter()
            .map(|v| v.into())
            .collect::<Vec<CachingIsotopicModel<'a>>>())
        .into()
    }
}

/// A single-shot deconvolution operation on the provided peak list.
///
///
/// # Arguments
/// - `peaks`: The centroided mass spectrum to process
/// - `isotopic_model`: The model to generate isotpoic patterns from, or a collection there-of
/// - `error_tolerance`: The mass accuracy constraint for isotopic peaks within a pattern
/// - `charge_range`: The minimum to maximum charge state to consider, ordered by absolute magnitude
/// - `scorer`: The strategy for scoring isotopic pattern fits
/// - `fit_filter`: The strategy for filtering out isotopic pattern fits that are too poor to consider
/// - `max_missed_peaks`: The number of missing isotopic peaks to tolerate in an isotopic pattern fit, regardless of quality
/// - `isotopic_params`: The set of parameters to use for `isotopic_model` when generating an isotopic pattern for a given m/z
/// - `use_quick_charge`: Whether or not to use Hoopman's QuickCharge algorithm to filter candidate charge states
///
/// # Note
/// If you are calling this function with the same parameters on many peak lists, it
/// may be preferable to create a [`DeconvolutionEngine`] and call its identically named method
/// to avoid repeatedly recomputing the same isotopic patterns, allowing them to be cached for
/// the lifetime of that instance.
///
/// Internally, this function creates a [`DeconvolutionEngine`], calls [`DeconvolutionEngine::deconvolute_peaks`],
/// and then returns the deconvolved peak list.
///
/// # See also
/// [`DeconvolutionEngine::deconvolute_peaks`]
pub fn deconvolute_peaks<
    'lifespan,
    C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
    I: Into<IsotopicModelLike<'lifespan>>,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
>(
    peaks: MZPeakSetType<C>,
    isotopic_model: I,
    error_tolerance: Tolerance,
    charge_range: ChargeRange,
    scorer: S,
    fit_filter: F,
    max_missed_peaks: u16,
    isotopic_params: IsotopicPatternParams,
    use_quick_charge: bool,
) -> Result<PeakSetVec<DeconvolvedSolutionPeak, mzpeaks::Mass>, DeconvolutionError> {
    let mut engine: DeconvolutionEngine<'_, C, S, F> = DeconvolutionEngine::new(
        isotopic_params,
        isotopic_model.into(),
        scorer,
        fit_filter,
        use_quick_charge,
    );

    engine.deconvolute_peaks(peaks, error_tolerance, charge_range, max_missed_peaks)
}

/// A deconvolution solution composed of the entire deconvoluted mass spectrum
/// as well as a list of deconvolution solutions for specific peaks.
#[derive(Debug, Default, Clone)]
pub struct PeaksAndTargets {
    /// The complete deconvolved mass spectrum
    pub deconvoluted_peaks: MassPeakSetType<DeconvolvedSolutionPeak>,
    /// Solutions to targeted peaks' isotopic deconvolution
    pub targets: Vec<Option<DeconvolvedSolutionPeak>>,
}

impl PeaksAndTargets {
    pub fn new(
        deconvoluted_peaks: MassPeakSetType<DeconvolvedSolutionPeak>,
        targets: Vec<Option<DeconvolvedSolutionPeak>>,
    ) -> Self {
        Self {
            deconvoluted_peaks,
            targets,
        }
    }
}

/// A single-shot deconvolution operation on the provided peak list with a set of priority targets
///
/// # Arguments
/// - `peaks`: The centroided mass spectrum to process
/// - `isotopic_model`: The model to generate isotpoic patterns from, or a collection there-of
/// - `error_tolerance`: The mass accuracy constraint for isotopic peaks within a pattern
/// - `charge_range`: The minimum to maximum charge state to consider, ordered by absolute magnitude
/// - `scorer`: The strategy for scoring isotopic pattern fits
/// - `fit_filter`: The strategy for filtering out isotopic pattern fits that are too poor to consider
/// - `max_missed_peaks`: The number of missing isotopic peaks to tolerate in an isotopic pattern fit, regardless of quality
/// - `isotopic_params`: The set of parameters to use for `isotopic_model` when generating an isotopic pattern for a given m/z
/// - `use_quick_charge`: Whether or not to use Hoopman's QuickCharge algorithm to filter candidate
/// - `targets`: A sequence of m/z values which the deconvolution machinery should track specifically
///
/// See the note on [`deconvolute_peaks`] about using a [`DeconvolutionEngine`] instead if called repeatedly.
///
/// # See also
/// [`DeconvolutionEngine::deconvolute_peaks_with_targets`]
pub fn deconvolute_peaks_with_targets<
    'lifespan,
    C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
    I: Into<IsotopicModelLike<'lifespan>>,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
>(
    peaks: MZPeakSetType<C>,
    isotopic_model: I,
    error_tolerance: Tolerance,
    charge_range: ChargeRange,
    scorer: S,
    fit_filter: F,
    max_missed_peaks: u16,
    isotopic_params: IsotopicPatternParams,
    use_quick_charge: bool,
    targets: &[f64],
) -> Result<PeaksAndTargets, DeconvolutionError> {
    let mut engine: DeconvolutionEngine<'_, C, S, F> = DeconvolutionEngine::new(
        isotopic_params,
        isotopic_model.into(),
        scorer,
        fit_filter,
        use_quick_charge,
    );

    engine.deconvolute_peaks_with_targets(
        peaks,
        error_tolerance,
        charge_range,
        max_missed_peaks,
        targets,
    )
}

#[derive(Debug, Clone)]
/// A state-manager for deconvolution with a preserved isotopic pattern model cache
/// and a consistent set of parameters and strategies. If multiple isotopic
/// models are used, a slightly different algorithm will be used.
///
/// The type definition is templated on multiple compile time strategies. This means the
/// scoring function is fixed at compile time.
///
/// Prefer using an instance of this type to repeatedly calling [`deconvolute_peaks`],
/// which internally creates an instance and then discards after it is done. This way
/// you can preserve the computed isotopic pattern cache for speed and consistency across
/// input spectra.
///
/// ## What is this isotopic pattern cache thing?
/// Internally, any isotopic models you provide will be wrapped in [`CachingIsotopicModel`]
/// if they are not an instance already. This implementation of [`IsotopicPatternGenerator`](crate::isotopic_model::IsotopicPatternGenerator)
/// that caches computed isotopic patterns so that repeated queries don't have to waste time recalculating patterns.
/// Queries that are also very, very close to a cached solution will also re-use that solution, so we provide
/// a [`DeconvolutionEngine::populate_isotopic_model_cache`] method for pre-filling the cache with a grid of
/// m/z by charge state values so that most experimental data doesn't bias what is already computed in the cache.
pub struct DeconvolutionEngine<
    'lifespan,
    C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
> {
    /// The set of parameters to use for `isotopic_model` when generating an isotopic pattern for a given m/z
    isotopic_params: IsotopicPatternParams,
    /// Whether or not to use Hoopman's QuickCharge algorithm to filter candidate charge states prior to the full
    /// isotopic pattern fitting process
    use_quick_charge: bool,
    /// The model to generate isotpoic patterns from, or a collection there-of. If more than one model is
    /// provided, a slightly different algorithm will be used.
    isotopic_model: Option<IsotopicModelLike<'lifespan>>,
    /// The strategy for scoring isotopic pattern fits
    scorer: Option<S>,
    /// The strategy for filtering out isotopic pattern fits that are too poor to consider
    fit_filter: Option<F>,
    peak_type: PhantomData<C>,
}

impl<
        'lifespan,
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > DeconvolutionEngine<'lifespan, C, S, F>
{
    /// Create a new [`DeconvolutionEngine`] with the associated strategies
    /// # Arguments
    /// - `isotopic_params`: The set of parameters to use for `isotopic_model` when generating an isotopic pattern for a given m/z
    /// - `isotopic_model`: The model to generate isotpoic patterns from, or a collection of them.
    /// - `scorer`: The strategy for scoring isotopic pattern fits
    /// - `fit_filter`: The strategy for filtering out isotopic pattern fits that are too poor to consider
    /// - `use_quick_charge`: Whether or not to use Hoopman's QuickCharge algorithm to filter candidate
    pub fn new<I: Into<IsotopicModelLike<'lifespan>>>(
        isotopic_params: IsotopicPatternParams,
        isotopic_model: I,
        scorer: S,
        fit_filter: F,
        use_quick_charge: bool,
    ) -> Self {
        Self {
            isotopic_params,
            isotopic_model: Some(isotopic_model.into()),
            scorer: Some(scorer),
            fit_filter: Some(fit_filter),
            use_quick_charge,
            peak_type: PhantomData,
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
                        self.isotopic_params,
                    );
                }
                IsotopicModelLike::MultipleModels(caches) => {
                    for cache in caches {
                        cache.populate_cache_params(
                            min_mz,
                            max_mz,
                            min_charge,
                            max_charge,
                            self.isotopic_params,
                        );
                    }
                }
            }
        }
    }

    /// Deconvolute the provided `peaks` to neutral mass, charge labeled peaks.
    ///
    /// # Arguments
    /// - `peaks`: The centroided mass spectrum to process
    /// - `error_tolerance`: The mass accuracy constraint for isotopic peaks within a pattern
    /// - `charge_range`: The minimum to maximum charge state to consider, ordered by absolute magnitude
    /// - `max_missed_peaks`: The number of missing isotopic peaks to tolerate in an isotopic pattern fit, regardless of quality
    ///
    /// # Note
    /// This may lead to slightly different solutions depending upon which pattern-seed m/zs are cached,
    /// so precalculating the cache with [`DeconvolutionEngine::populate_isotopic_model_cache`] should
    /// be used for consistency.
    pub fn deconvolute_peaks(
        &mut self,
        peaks: MZPeakSetType<C>,
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        max_missed_peaks: u16,
    ) -> Result<MassPeakSetType<DeconvolvedSolutionPeak>, DeconvolutionError> {

        match mem::take(&mut self.isotopic_model).unwrap() {
            IsotopicModelLike::SingleModel(model) => {
                let mut deconvoluter =
                    GraphDeconvoluterType::<C, CachingIsotopicModel<'lifespan>, S, F>::new(
                        peaks,
                        model,
                        mem::take(&mut self.scorer).unwrap(),
                        mem::take(&mut self.fit_filter).unwrap(),
                        max_missed_peaks,
                        self.use_quick_charge,
                    );

                let output = deconvoluter.deconvolve(
                    error_tolerance,
                    charge_range,
                    1,
                    0,
                    self.isotopic_params,
                    1e-3,
                    10,
                );

                self.isotopic_model = Some(deconvoluter.inner.isotopic_model.into());
                self.scorer = Some(deconvoluter.inner.scorer);
                self.fit_filter = Some(deconvoluter.inner.fit_filter);
                output
            }
            IsotopicModelLike::MultipleModels(models) => {
                let mut deconvoluter =
                    GraphMultiDeconvoluterType::<C, CachingIsotopicModel<'lifespan>, S, F>::new(
                        peaks,
                        models,
                        mem::take(&mut self.scorer).unwrap(),
                        mem::take(&mut self.fit_filter).unwrap(),
                        max_missed_peaks,
                        self.use_quick_charge,
                    );

                let output = deconvoluter.deconvolve(
                    error_tolerance,
                    charge_range,
                    1,
                    0,
                    self.isotopic_params,
                    1e-3,
                    10,
                );

                self.isotopic_model = Some(deconvoluter.inner.isotopic_models.into());
                self.scorer = Some(deconvoluter.inner.scorer);
                self.fit_filter = Some(deconvoluter.inner.fit_filter);
                output
            }
        }
    }

    /// Deconvolute the provided `peaks` to neutral mass, charge labeled peaks with a set of priority targets.
    ///
    /// See the note on [`DeconvolutionEngine::deconvolute_peaks`] regarding cache seeding.
    ///
    /// # Arguments
    /// - `peaks`: The centroided mass spectrum to process
    /// - `error_tolerance`: The mass accuracy constraint for isotopic peaks within a pattern
    /// - `charge_range`: The minimum to maximum charge state to consider, ordered by absolute magnitude
    /// - `max_missed_peaks`: The number of missing isotopic peaks to tolerate in an isotopic pattern fit, regardless of quality
    /// - `targets`: A sequence of m/z values which the deconvolution machinery should track specifically
    pub fn deconvolute_peaks_with_targets(
        &mut self,
        peaks: MZPeakSetType<C>,
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        max_missed_peaks: u16,
        targets: &[f64],
    ) -> Result<PeaksAndTargets, DeconvolutionError> {
        let output = match mem::take(&mut self.isotopic_model).unwrap() {
            IsotopicModelLike::SingleModel(model) => {
                let mut deconvoluter =
                    GraphDeconvoluterType::<C, CachingIsotopicModel<'lifespan>, S, F>::new(
                        peaks,
                        model,
                        mem::take(&mut self.scorer).unwrap(),
                        mem::take(&mut self.fit_filter).unwrap(),
                        max_missed_peaks,
                        self.use_quick_charge,
                    );

                let links: Vec<_> = targets
                    .iter()
                    .map(|mz| {
                        let peak = deconvoluter.has_peak(*mz, error_tolerance);
                        deconvoluter.targeted_deconvolution(
                            peak,
                            error_tolerance,
                            charge_range,
                            1,
                            1,
                            self.isotopic_params,
                        )
                    })
                    .collect();

                let deconvoluted_peaks = deconvoluter.deconvolve(
                    error_tolerance,
                    charge_range,
                    1,
                    0,
                    self.isotopic_params,
                    1e-3,
                    10,
                )?;

                let targets: Vec<Option<DeconvolvedSolutionPeak>> = links
                    .into_iter()
                    .map(|target| {
                        deconvoluter
                            .resolve_target(&deconvoluted_peaks, &target)
                            .cloned()
                    })
                    .collect();

                self.isotopic_model = Some(deconvoluter.inner.isotopic_model.into());
                self.scorer = Some(deconvoluter.inner.scorer);
                self.fit_filter = Some(deconvoluter.inner.fit_filter);
                Ok(PeaksAndTargets::new(deconvoluted_peaks, targets))
            }
            IsotopicModelLike::MultipleModels(models) => {
                let mut deconvoluter =
                    GraphMultiDeconvoluterType::<C, CachingIsotopicModel<'lifespan>, S, F>::new(
                        peaks,
                        models,
                        mem::take(&mut self.scorer).unwrap(),
                        mem::take(&mut self.fit_filter).unwrap(),
                        max_missed_peaks,
                        self.use_quick_charge,
                    );

                let links: Vec<_> = targets
                    .iter()
                    .map(|mz| {
                        let peak = deconvoluter.has_peak(*mz, error_tolerance);
                        deconvoluter.targeted_deconvolution(
                            peak,
                            error_tolerance,
                            charge_range,
                            1,
                            1,
                            self.isotopic_params,
                        )
                    })
                    .collect();

                let deconvoluted_peaks = deconvoluter.deconvolve(
                    error_tolerance,
                    charge_range,
                    1,
                    0,
                    self.isotopic_params,
                    1e-3,
                    10,
                )?;

                let targets: Vec<Option<DeconvolvedSolutionPeak>> = links
                    .into_iter()
                    .map(|target| {
                        deconvoluter
                            .resolve_target(&deconvoluted_peaks, &target)
                            .cloned()
                    })
                    .collect();

                self.isotopic_model = Some(deconvoluter.inner.isotopic_models.into());
                self.scorer = Some(deconvoluter.inner.scorer);
                self.fit_filter = Some(deconvoluter.inner.fit_filter);
                Ok(PeaksAndTargets::new(deconvoluted_peaks, targets))
            }
        };
        output
    }

    /// The set of parameters to use for `isotopic_model` when generating an isotopic pattern for a given m/z
    pub fn isotopic_params(&self) -> &IsotopicPatternParams {
        &self.isotopic_params
    }

    /// Whether or not to use Hoopman's QuickCharge algorithm to filter candidate charge states prior to the full
    /// isotopic pattern fitting process
    pub fn use_quick_charge(&self) -> bool {
        self.use_quick_charge
    }

    pub fn set_use_quick_charge(&mut self, value: bool) {
        self.use_quick_charge = value;
    }

    /// The strategy for scoring isotopic pattern fits
    pub fn scorer(&self) -> Option<&S> {
        self.scorer.as_ref()
    }

    pub fn scorer_mut(&mut self) -> &mut Option<S> {
        &mut self.scorer
    }

    /// The strategy for filtering out isotopic pattern fits that are too poor to consider
    pub fn fit_filter(&self) -> Option<&F> {
        self.fit_filter.as_ref()
    }

    pub fn fit_filter_mut(&mut self) -> &mut Option<F> {
        &mut self.fit_filter
    }

    /// The model(s) to generate isotpoic patterns from. If more than one model is
    /// provided, a slightly different algorithm will be used.
    pub fn isotopic_model(&self) -> Option<&IsotopicModelLike<'lifespan>> {
        self.isotopic_model.as_ref()
    }

}
