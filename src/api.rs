use std::marker::PhantomData;
use std::mem;

use mzpeaks::{
    peak_set::PeakSetVec, CentroidLike, CentroidPeak, IntensityMeasurementMut, MZPeakSetType,
    MassPeakSetType, Tolerance,
};

use crate::{
    charge::ChargeRange,
    deconv_traits::{IsotopicDeconvolutionAlgorithm, IsotopicPatternFitter, TargetedDeconvolution},
    deconvoluter::GraphDeconvoluterType,
    isotopic_model::{CachingIsotopicModel, IsotopicPatternParams},
    scorer::{IsotopicFitFilter, IsotopicPatternScorer},
    solution::DeconvolvedSolutionPeak,
};

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
) -> PeakSetVec<DeconvolvedSolutionPeak, mzpeaks::Mass> {
    let mut deconvoluter = GraphDeconvoluterType::<C, CachingIsotopicModel<'lifespan>, S, F>::new(
        peaks,
        isotopic_model.into(),
        scorer,
        fit_filter,
        max_missed_peaks,
    );

    let output = deconvoluter.deconvolve(
        error_tolerance,
        charge_range,
        1,
        0,
        isotopic_params,
        1e-3,
        10,
    );
    output
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
    targets: &[f64],
) -> PeaksAndTargets {
    let mut deconvoluter = GraphDeconvoluterType::<C, CachingIsotopicModel<'lifespan>, S, F>::new(
        peaks,
        isotopic_model.into(),
        scorer,
        fit_filter,
        max_missed_peaks,
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
                isotopic_params,
            )
        })
        .collect();

    let deconvoluted_peaks = deconvoluter.deconvolve(
        error_tolerance,
        charge_range,
        1,
        0,
        isotopic_params,
        1e-3,
        10,
    );

    let targets: Vec<Option<DeconvolvedSolutionPeak>> = links
        .into_iter()
        .map(|target| {
            deconvoluter
                .resolve_target(&deconvoluted_peaks, &target)
                .cloned()
        })
        .collect();
    PeaksAndTargets::new(deconvoluted_peaks, targets)
}

#[derive(Debug)]
pub struct DeconvolutionEngine<
    'lifespan,
    C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
> {
    isotopic_params: IsotopicPatternParams,
    isotopic_model: Option<CachingIsotopicModel<'lifespan>>,
    scorer: Option<S>,
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
    pub fn new(
        isotopic_params: IsotopicPatternParams,
        isotopic_model: CachingIsotopicModel<'lifespan>,
        scorer: S,
        fit_filter: F,
    ) -> Self {
        Self {
            isotopic_params,
            isotopic_model: Some(isotopic_model),
            scorer: Some(scorer),
            fit_filter: Some(fit_filter),
            peak_type: PhantomData,
        }
    }

    pub fn populate_isotopic_model_cache(&mut self, min_mz: f64, max_mz: f64, min_charge: i32, max_charge: i32) {
        if let Some(cache) = self.isotopic_model.as_mut() {
            cache.populate_cache_params(min_mz, max_mz, min_charge, max_charge, self.isotopic_params);
        }
    }

    pub fn deconvolute_peaks(
        &mut self,
        peaks: MZPeakSetType<C>,
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        max_missed_peaks: u16,
    ) -> MassPeakSetType<DeconvolvedSolutionPeak> {
        let mut deconvoluter =
            GraphDeconvoluterType::<C, CachingIsotopicModel<'lifespan>, S, F>::new(
                peaks,
                mem::take(&mut self.isotopic_model).unwrap(),
                mem::take(&mut self.scorer).unwrap(),
                mem::take(&mut self.fit_filter).unwrap(),
                max_missed_peaks,
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

    pub fn deconvolute_peaks_with_targets(
        &mut self,
        peaks: MZPeakSetType<C>,
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        max_missed_peaks: u16,
        targets: &[f64],
    ) -> PeaksAndTargets {
        let mut deconvoluter =
            GraphDeconvoluterType::<C, CachingIsotopicModel<'lifespan>, S, F>::new(
                peaks,
                mem::take(&mut self.isotopic_model).unwrap(),
                mem::take(&mut self.scorer).unwrap(),
                mem::take(&mut self.fit_filter).unwrap(),
                max_missed_peaks,
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
        );

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

        PeaksAndTargets::new(deconvoluted_peaks, targets)
    }
}
