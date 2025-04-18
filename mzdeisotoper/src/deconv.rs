use std::collections::HashMap;

use itertools::Itertools;

use mzdata::{io::MassSpectrometryFormat, prelude::*, spectrum::{ScanPolarity, SignalContinuity}, Param};
use mzpeaks::MZPeakSetType;

use mzdeisotope::{
    isolation::{Coisolation, PrecursorPurityEstimator},
    scorer::{IsotopicFitFilter, IsotopicPatternScorer},
    solution::DeconvolvedSolutionPeak,
    DeconvolutionEngine, PeaksAndTargets,
};

use crate::{
    args::{DeconvolutionParams, PrecursorProcessing, SignalParams},
    progress::ProgressRecord,
    types::{CPeak, SpectrumGroupType, SpectrumType, PEAK_COUNT_THRESHOLD_WARNING},
    write::postprocess_spectra,
};

pub fn coisolation_to_param(c: &Coisolation) -> Param {
    Param::new_key_value(
        "mzdeisotope:coisolation".to_string(),
        format!(
            "{} {} {}",
            c.neutral_mass,
            c.intensity,
            c.charge.unwrap_or_default()
        ),
    )
}

pub struct PrecursorPeakMapper<'a> {
    mzs: &'a [f64],
    peaks: &'a [Option<DeconvolvedSolutionPeak>],
}

impl<'a> PrecursorPeakMapper<'a> {
    pub fn new(mzs: &'a [f64], peaks: &'a [Option<DeconvolvedSolutionPeak>]) -> Self {
        Self { mzs, peaks }
    }

    pub fn find_peak_for_mz(&self, mz: f64) -> Option<&DeconvolvedSolutionPeak> {
        self.find_mz(mz).and_then(|i| self.peaks[i].as_ref())
    }

    pub fn find_mz(&self, mz: f64) -> Option<usize> {
        self.mzs
            .iter()
            .find_position(|m| (**m - mz).abs() < 1e-6)
            .map(|(i, _)| i)
    }
}

pub fn purities_of(
    purity_estimator: &PrecursorPurityEstimator,
    group: &SpectrumGroupType,
    targets: &PrecursorPeakMapper,
    is_dia: bool,
) -> HashMap<usize, (f32, Vec<Coisolation>)> {
    let mut purities = HashMap::new();
    if let Some(precursor_scan) = group.precursor() {
        group.products().iter().enumerate().for_each(|(i, scan)| {
            if let Some(prec) = scan.precursor() {
                if is_dia {
                    let coisolations = purity_estimator.coisolation(
                        precursor_scan.deconvoluted_peaks.as_ref().unwrap(),
                        &DeconvolvedSolutionPeak::new(
                            prec.ion().mz,
                            0.0,
                            1,
                            0,
                            0.0,
                            Box::default(),
                        ),
                        Some(&prec.isolation_window),
                        0.1,
                        true,
                    );
                    purities.insert(i, (0.0, coisolations));
                } else if let Some(peak) = targets.find_peak_for_mz(prec.ion().mz) {
                    if is_dia {
                        // For DIA mode, purity isn't meaningful
                    } else {
                        let purity = purity_estimator.precursor_purity(
                            precursor_scan.peaks.as_ref().unwrap(),
                            peak,
                            Some(&prec.isolation_window),
                        );
                        let coisolations = purity_estimator.coisolation(
                            precursor_scan.deconvoluted_peaks.as_ref().unwrap(),
                            peak,
                            Some(&prec.isolation_window),
                            0.1,
                            true,
                        );
                        purities.insert(i, (purity, coisolations));
                    }
                }
            }
        });
    }
    purities
}

#[tracing::instrument(
    level = "debug",
    skip(
        scan,
        precursor_processing,
        selected_mz_ranges,
        signal_processing_params
    )
)]
pub fn pick_ms1_peaks(
    scan: &mut SpectrumType,
    precursor_processing: &PrecursorProcessing,
    selected_mz_ranges: &[(f64, f64)],
    signal_processing_params: &SignalParams,
) -> Option<MZPeakSetType<CPeak>> {
    match scan.signal_continuity() {
        SignalContinuity::Unknown => {
            panic!("Can't infer peak mode for {}", scan.id())
        }
        SignalContinuity::Centroid => match precursor_processing {
            PrecursorProcessing::Full | PrecursorProcessing::MS1Only | PrecursorProcessing::DIA => {
                Some(scan.try_build_centroids().unwrap().clone())
            }
            PrecursorProcessing::SelectedPrecursors => {
                let peaks = scan.try_build_centroids().unwrap();
                let peaks = selected_mz_ranges
                    .iter()
                    .flat_map(|(low, high)| peaks.between(*low, *high, Tolerance::PPM(5.0)))
                    .cloned()
                    .collect();

                Some(peaks)
            }
            PrecursorProcessing::TandemOnly => None,
        },
        SignalContinuity::Profile => {
            if signal_processing_params.ms1_denoising > 0.0 {
                tracing::trace!("Denoising {}", scan.id());
                if let Err(e) = scan.denoise(signal_processing_params.ms1_denoising) {
                    tracing::error!("An error occurred while denoising {}: {e}", scan.id());
                }
            }
            match precursor_processing {
                PrecursorProcessing::SelectedPrecursors => {
                    scan.pick_peaks_in_intervals(1.0, selected_mz_ranges)
                        .unwrap();
                    scan.description_mut().signal_continuity = SignalContinuity::Centroid;
                    Some(scan.peaks.clone().unwrap())
                }
                PrecursorProcessing::Full
                | PrecursorProcessing::MS1Only
                | PrecursorProcessing::DIA => {
                    scan.pick_peaks(1.0).unwrap();
                    scan.description_mut().signal_continuity = SignalContinuity::Centroid;
                    Some(scan.peaks.clone().unwrap())
                }
                PrecursorProcessing::TandemOnly => None,
            }
        }
    }
}

pub fn pick_msn_peaks(
    scan: &mut SpectrumType,
    _signal_processing_params: &SignalParams,
) -> MZPeakSetType<CPeak> {
    match scan.signal_continuity() {
        SignalContinuity::Unknown => {
            panic!("Can't infer peak mode for {}", scan.id())
        }
        SignalContinuity::Centroid => scan.try_build_centroids().unwrap().clone(),
        SignalContinuity::Profile => {
            scan.pick_peaks(1.0).unwrap();
            scan.description_mut().signal_continuity = SignalContinuity::Centroid;
            scan.peaks.clone().unwrap()
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn deconvolution_transform<
    S: IsotopicPatternScorer + Send + 'static,
    F: IsotopicFitFilter + Send + 'static,
    SN: IsotopicPatternScorer + Send + 'static,
    FN: IsotopicFitFilter + Send + 'static,
>(
    ms1_engine: &mut DeconvolutionEngine<'_, CPeak, S, F>,
    msn_engine: &mut DeconvolutionEngine<'_, CPeak, SN, FN>,
    ms1_deconv_params: &DeconvolutionParams,
    msn_deconv_params: &DeconvolutionParams,
    signal_processing_params: &SignalParams,
    group_idx: usize,
    mut group: SpectrumGroupType,
    precursor_processing: PrecursorProcessing,
    writer_format: MassSpectrometryFormat,
) -> (usize, SpectrumGroupType, ProgressRecord) {
    let had_precursor = group.precursor().is_some();
    let mut prog = ProgressRecord::default();
    let is_dia = matches!(precursor_processing, PrecursorProcessing::DIA);

    let precursor_mz: Vec<_> = if is_dia {
        vec![]
    } else {
        group
            .products()
            .iter()
            .flat_map(|s| s.precursor().map(|prec| prec.ion().mz))
            .collect()
    };

    let purity_estimator = PrecursorPurityEstimator::default();

    let selected_mz_ranges = group.selected_intervals(1.0, 3.0);

    // Process the precursor spectrum and collect the updated selected ions' solutions
    let targets = match group.precursor_mut() {
        Some(scan) => {
            if tracing::enabled!(tracing::Level::TRACE) {
                tracing::trace!(
                    "Processing {} MS{} ({:0.3})",
                    scan.id(),
                    scan.ms_level(),
                    scan.start_time()
                );
            }

            let peaks = pick_ms1_peaks(
                scan,
                &precursor_processing,
                &selected_mz_ranges,
                signal_processing_params,
            );

            let charge_range = match scan.polarity() {
                ScanPolarity::Unknown
                | ScanPolarity::Positive => {
                    let (mut low, mut high) = ms1_deconv_params.charge_range;
                    low = low.abs();
                    high = high.abs();
                    (low.min(high), high.max(low))
                }
                ScanPolarity::Negative => {
                    let (mut low, mut high) = ms1_deconv_params.charge_range;
                    low = -low.abs();
                    high = -high.abs();
                    (low.min(high), high.max(low))
                }
            };

            if let Some(peaks) = peaks {
                let has_too_many_peaks = peaks.len() >= PEAK_COUNT_THRESHOLD_WARNING;
                if has_too_many_peaks {
                    tracing::warn!("{} has {} centroids", scan.id(), peaks.len())
                }
                let PeaksAndTargets {
                    deconvoluted_peaks,
                    targets,
                } = ms1_engine
                    .deconvolute_peaks_with_targets(
                        peaks,
                        Tolerance::PPM(20.0),
                        charge_range,
                        ms1_deconv_params.max_missed_peaks,
                        &precursor_mz,
                    )
                    .unwrap();
                if has_too_many_peaks {
                    tracing::warn!(
                        "{} has {} deconvolved centroids",
                        scan.id(),
                        deconvoluted_peaks.len()
                    )
                }
                prog.ms1_peaks = deconvoluted_peaks.len();
                prog.ms1_spectra += 1;
                scan.deconvoluted_peaks = Some(deconvoluted_peaks);
                targets
            } else {
                // We aren't actually deconvoluting the MS1 spectrum, so create a stub for each precursor
                vec![None; precursor_mz.len()]
            }
        }
        None => vec![None; precursor_mz.len()],
    };
    let mut purities = HashMap::new();

    let precursor_map = PrecursorPeakMapper::new(&precursor_mz, &targets);

    if had_precursor {
        purities = purities_of(&purity_estimator, &group, &precursor_map, is_dia);
    }

    group
        .products_mut()
        .iter_mut()
        .enumerate()
        .for_each(|(scan_i, scan)| {
            if !had_precursor && tracing::enabled!(tracing::Level::TRACE) {
                tracing::trace!(
                    "Processing {} MS{} ({:0.3})",
                    scan.id(),
                    scan.ms_level(),
                    scan.acquisition().start_time()
                );
            }

            let precursor_charge = scan
                .precursor()
                .and_then(|prec| prec.charge())
                .unwrap_or(msn_deconv_params.charge_range.1);

            let mut msn_charge_range = msn_deconv_params.charge_range;

            if precursor_charge.abs() < msn_charge_range.1.abs() && precursor_charge.abs() != 0 {
                msn_charge_range.1 = precursor_charge;
            }

            msn_charge_range = match scan.polarity() {
                ScanPolarity::Unknown
                | ScanPolarity::Positive => {
                    let (mut low, mut high) = msn_charge_range;
                    low = low.abs();
                    high = high.abs();
                    (low.min(high), high.max(low))
                }
                ScanPolarity::Negative => {
                    let (mut low, mut high) = msn_charge_range;
                    low = -low.abs();
                    high = -high.abs();
                    (low.min(high), high.max(low))
                }
            };

            let peaks = pick_msn_peaks(scan, signal_processing_params);
            {
                let deconvoluted_peaks = msn_engine
                    .deconvolute_peaks(
                        peaks,
                        Tolerance::PPM(20.0),
                        msn_charge_range,
                        msn_deconv_params.max_missed_peaks,
                    )
                    .unwrap();
                prog.msn_peaks += deconvoluted_peaks.len();
                prog.msn_spectra += 1;
                scan.deconvoluted_peaks = Some(deconvoluted_peaks);
            }

            if is_dia {
                if let Some(prec) = scan.precursor_mut() {
                    let (_, coisolated) = purities.remove(&scan_i).unwrap_or_default();
                    coisolated.iter().for_each(|c| {
                        prec.ion_mut().params_mut().push(coisolation_to_param(c));
                    });
                }
            } else if let Some(prec) = scan.precursor_mut() {
                let target_mz = prec.mz();
                let _ = precursor_map
                    .find_mz(target_mz)
                    .map(|i| {
                        if let Some(peak) = &targets[i] {
                            let orig_charge = prec.ion().charge;
                            let update_ion = if let Some(orig_z) = orig_charge {
                                let t = orig_z == peak.charge;
                                if !t {
                                    prog.precursor_charge_state_mismatch += 1;
                                }
                                t
                            } else {
                                true
                            };
                            let prec_ion = prec.ion_mut();
                            if update_ion {
                                prec_ion.mz = peak.mz();
                                prec_ion.charge = Some(peak.charge);
                                prec_ion.intensity = peak.intensity;
                                let (purity, coisolated) =
                                    purities.remove(&scan_i).unwrap_or_default();
                                prec_ion.params_mut().push(Param::new_key_value(
                                    "mzdeisotope:isolation purity".to_string(),
                                    purity.to_string(),
                                ));
                                coisolated.iter().for_each(|c| {
                                    prec_ion.params_mut().push(coisolation_to_param(c));
                                });
                            } else {
                                prec_ion.params_mut().push(Param::new_key_value(
                                    "mzdeisotope:defaulted".to_string(),
                                    true.to_string(),
                                ));
                                let (purity, coisolated) =
                                    purities.remove(&scan_i).unwrap_or_default();
                                prec_ion.params_mut().push(Param::new_key_value(
                                    "mzdeisotope:isolation purity".to_string(),
                                    purity.to_string(),
                                ));
                                coisolated.iter().for_each(|c| {
                                    prec_ion.params_mut().push(coisolation_to_param(c));
                                });
                                prog.precursors_defaulted += 1;
                            }
                        }
                    })
                    .or_else(|| {
                        let prec_ion = prec.ion_mut();
                        prec_ion.params_mut().push(Param::new_key_value(
                            "mzdeisotope:defaulted".to_string(),
                            true.to_string(),
                        ));
                        prec_ion.params_mut().push(Param::new_key_value(
                            "mzdeisotope:orphan".to_string(),
                            true.to_string(),
                        ));
                        prog.precursors_defaulted += 1;
                        None
                    });
            }
        });

    let (group_idx, group) = postprocess_spectra(group_idx, group, writer_format);
    (group_idx, group, prog)
}
