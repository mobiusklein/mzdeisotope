/*!
 * High level APIs for running deconvolution operations
 */

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
    scorer::{IsotopicFitFilter, IsotopicPatternScorer},
    solution::DeconvolvedSolutionPeak,
};

/// A single-shot deconvolution operation on the provided peak list
///
/// # Arguments
/// - `peaks`: The centroided mass spectrum to process
/// - `isotopic_model`: The model to generate isotpoic patterns from
/// - `error_tolerance`: The mass accuracy constraint for isotopic peaks within a pattern
/// - `charge_range`: The minimum to maximum charge state to consider, ordered by absolute magnitude
/// - `scorer`: The strategy for scoring isotopic pattern fits
/// - `fit_filter`: The strategy for filtering out isotopic pattern fits that are too poor to consider
/// - `max_missed_peaks`: The number of missing isotopic peaks to tolerate in an isotopic pattern fit, regardless of quality
/// - `isotopic_params`: The set of parameters to use for `isotopic_model` when generating an isotopic pattern for a given m/z
/// - `use_quick_charge`: Whether or not to use Hoopman's QuickCharge algorithm to filter candidate charge states
pub fn deconvolute_peaks<
    'lifespan,
    C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
    I: Into<CachingIsotopicModel<'lifespan>>,
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

#[derive(Debug, Default, Clone)]
pub struct PeaksAndTargets {
    pub deconvoluted_peaks: MassPeakSetType<DeconvolvedSolutionPeak>,
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
/// - `isotopic_model`: The model to generate isotpoic patterns from
/// - `error_tolerance`: The mass accuracy constraint for isotopic peaks within a pattern
/// - `charge_range`: The minimum to maximum charge state to consider, ordered by absolute magnitude
/// - `scorer`: The strategy for scoring isotopic pattern fits
/// - `fit_filter`: The strategy for filtering out isotopic pattern fits that are too poor to consider
/// - `max_missed_peaks`: The number of missing isotopic peaks to tolerate in an isotopic pattern fit, regardless of quality
/// - `isotopic_params`: The set of parameters to use for `isotopic_model` when generating an isotopic pattern for a given m/z
/// - `use_quick_charge`: Whether or not to use Hoopman's QuickCharge algorithm to filter candidate
/// - `targets`: A sequence of m/z values which the deconvolution machinery should track specifically
pub fn deconvolute_peaks_with_targets<
    'lifespan,
    C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
    I: Into<CachingIsotopicModel<'lifespan>>,
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
/// and a consistent set of parameters and strategies.
///
/// The type definition is templated on multiple compile time strategies.
///
/// Internally, this struct is in a mostly-unusable state while it is running a deconvolution
/// operation, as gives up its ownership of its isotopic pattern cache temporarily. The cache is
/// reclaimed after processing finishes.
pub struct DeconvolutionEngine<
    'lifespan,
    C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
> {
    /// The set of parameters to use for `isotopic_model` when generating an isotopic pattern for a given m/z
    isotopic_params: IsotopicPatternParams,
    /// Whether or not to use Hoopman's QuickCharge algorithm to filter candidate
    use_quick_charge: bool,
    /// The model to generate isotpoic patterns from
    isotopic_model: Option<CachingIsotopicModel<'lifespan>>,
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
    /// - `isotopic_model`: The model to generate isotpoic patterns from
    /// - `scorer`: The strategy for scoring isotopic pattern fits
    /// - `fit_filter`: The strategy for filtering out isotopic pattern fits that are too poor to consider
    /// - `use_quick_charge`: Whether or not to use Hoopman's QuickCharge algorithm to filter candidate
    pub fn new(
        isotopic_params: IsotopicPatternParams,
        isotopic_model: CachingIsotopicModel<'lifespan>,
        scorer: S,
        fit_filter: F,
        use_quick_charge: bool,
    ) -> Self {
        Self {
            isotopic_params,
            isotopic_model: Some(isotopic_model),
            scorer: Some(scorer),
            fit_filter: Some(fit_filter),
            use_quick_charge,
            peak_type: PhantomData,
        }
    }

    /// Pre-calculcate and cache all isotopic patterns between `min_mz` and `max_mz` for
    /// charge states between `min_charge` and `max_charge`.
    pub fn populate_isotopic_model_cache(
        &mut self,
        min_mz: f64,
        max_mz: f64,
        min_charge: i32,
        max_charge: i32,
    ) {
        if let Some(cache) = self.isotopic_model.as_mut() {
            cache.populate_cache_params(
                min_mz,
                max_mz,
                min_charge,
                max_charge,
                self.isotopic_params,
            );
        }
    }

    /// Deconvolute the provided `peaks` to neutral mass, charge labeled peaks
    /// # Arguments
    /// - `peaks`: The centroided mass spectrum to process
    /// - `error_tolerance`: The mass accuracy constraint for isotopic peaks within a pattern
    /// - `charge_range`: The minimum to maximum charge state to consider, ordered by absolute magnitude
    /// - `max_missed_peaks`: The number of missing isotopic peaks to tolerate in an isotopic pattern fit, regardless of quality
    pub fn deconvolute_peaks(
        &mut self,
        peaks: MZPeakSetType<C>,
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        max_missed_peaks: u16,
    ) -> Result<MassPeakSetType<DeconvolvedSolutionPeak>, DeconvolutionError> {
        let mut deconvoluter =
            GraphDeconvoluterType::<C, CachingIsotopicModel<'lifespan>, S, F>::new(
                peaks,
                mem::take(&mut self.isotopic_model).unwrap(),
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

        self.isotopic_model = Some(deconvoluter.inner.isotopic_model);
        self.scorer = Some(deconvoluter.inner.scorer);
        self.fit_filter = Some(deconvoluter.inner.fit_filter);

        output
    }

    /// Deconvolute the provided `peaks` to neutral mass, charge labeled peaks with a set of priority targets
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
        let mut deconvoluter =
            GraphDeconvoluterType::<C, CachingIsotopicModel<'lifespan>, S, F>::new(
                peaks,
                mem::take(&mut self.isotopic_model).unwrap(),
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

        self.isotopic_model = Some(deconvoluter.inner.isotopic_model);
        self.scorer = Some(deconvoluter.inner.scorer);
        self.fit_filter = Some(deconvoluter.inner.fit_filter);

        Ok(PeaksAndTargets::new(deconvoluted_peaks, targets))
    }
}
