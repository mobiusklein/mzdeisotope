use std::borrow::Cow;

use itertools::Itertools;

use mzdata::{prelude::*, spectrum::SignalContinuity, Param};
use mzpeaks::prelude::*;

use mzdeisotope::{
    api::{DeconvolutionEngine, PeaksAndTargets},
    scorer::{IsotopicFitFilter, IsotopicPatternScorer},
};

use crate::{
    args::{DeconvolutionParams, PrecursorProcessing, SignalParams},
    progress::ProgressRecord,
    types::{CPeak, SpectrumGroupType},
};

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
) -> (usize, SpectrumGroupType, ProgressRecord) {
    let had_precursor = group.precursor().is_some();
    let mut prog = ProgressRecord::default();

    let precursor_mz: Vec<_> = group
        .products()
        .into_iter()
        .flat_map(|s| s.precursor().and_then(|prec| Some(prec.ion().mz)))
        .collect();

    let selected_mz_ranges = group.selected_intervals(1.0, 3.0);
    let targets = match group.precursor_mut() {
        Some(scan) => {
            log::trace!(
                "Processing {} MS{} ({:0.3})",
                scan.id(),
                scan.ms_level(),
                scan.acquisition().start_time()
            );
            let peaks = match scan.signal_continuity() {
                SignalContinuity::Unknown => {
                    panic!("Can't infer peak mode for {}", scan.id())
                }
                SignalContinuity::Centroid => {
                    match precursor_processing {
                        PrecursorProcessing::Full | PrecursorProcessing::MS1Only => {
                            Some(Cow::Borrowed(scan.try_build_centroids().unwrap()))
                        },
                        PrecursorProcessing::SelectedPrecursors => {
                            let peaks = scan.try_build_centroids().unwrap();
                            let peaks = selected_mz_ranges.iter().map(|(low, high)| {
                                peaks.between(*low, *high, Tolerance::PPM(5.0))
                            }).flatten().cloned().collect();

                            Some(Cow::Owned(peaks))
                        },
                        PrecursorProcessing::TandemOnly => None,
                    }
                },
                SignalContinuity::Profile => {
                    if signal_processing_params.ms1_denoising > 0.0 {
                        log::trace!("Denoising {}", scan.id());
                        if let Err(e) = scan.denoise(signal_processing_params.ms1_denoising) {
                            log::error!("An error occurred while denoising {}: {e}", scan.id());
                        }
                    }
                    match precursor_processing {
                        PrecursorProcessing::SelectedPrecursors => {
                            scan.pick_peaks_in_intervals(
                                1.0,
                                Default::default(),
                                &selected_mz_ranges,
                            )
                            .unwrap();
                            scan.description_mut().signal_continuity = SignalContinuity::Centroid;
                            Some(Cow::Borrowed(scan.peaks.as_ref().unwrap()))
                        }
                        PrecursorProcessing::Full | PrecursorProcessing::MS1Only => {
                            scan.pick_peaks(1.0, Default::default()).unwrap();
                            scan.description_mut().signal_continuity = SignalContinuity::Centroid;
                            Some(Cow::Borrowed(scan.peaks.as_ref().unwrap()))
                        }
                        PrecursorProcessing::TandemOnly => None,
                    }
                }
            };

            if let Some(peaks) = peaks {
                let PeaksAndTargets {
                    deconvoluted_peaks,
                    targets,
                } = ms1_engine
                    .deconvolute_peaks_with_targets(
                        match peaks {
                            Cow::Borrowed(x) => x.clone(),
                            Cow::Owned(x) => x,
                        },
                        Tolerance::PPM(20.0),
                        ms1_deconv_params.charge_range,
                        ms1_deconv_params.max_missed_peaks,
                        &precursor_mz,
                    )
                    .unwrap();
                prog.ms1_peaks = deconvoluted_peaks.len();
                prog.ms1_spectra += 1;
                scan.deconvoluted_peaks = Some(deconvoluted_peaks);
                targets
            } else {
                Vec::new()
            }
        }
        None => precursor_mz.iter().map(|_| None).collect(),
    };

    group.products_mut().iter_mut().for_each(|scan| {
        if !had_precursor {
            log::trace!(
                "Processing {} MS{} ({:0.3})",
                scan.id(),
                scan.ms_level(),
                scan.acquisition().start_time()
            );
        }

        let precursor_charge = scan
            .precursor()
            .and_then(|prec| prec.charge())
            .unwrap_or_else(|| msn_deconv_params.charge_range.1);

        let mut msn_charge_range = msn_deconv_params.charge_range;
        msn_charge_range.1 = msn_charge_range.1.max(precursor_charge);

        let peaks = match scan.signal_continuity() {
            SignalContinuity::Unknown => {
                panic!("Can't infer peak mode for {}", scan.id())
            }
            SignalContinuity::Centroid => scan.try_build_centroids().unwrap(),
            SignalContinuity::Profile => {
                scan.pick_peaks(1.0, Default::default()).unwrap();
                scan.description_mut().signal_continuity = SignalContinuity::Centroid;
                scan.peaks.as_ref().unwrap()
            }
        };

        let deconvoluted_peaks = msn_engine
            .deconvolute_peaks(
                peaks.clone(),
                Tolerance::PPM(20.0),
                msn_charge_range,
                msn_deconv_params.max_missed_peaks,
            )
            .unwrap();
        prog.msn_peaks += deconvoluted_peaks.len();
        prog.msn_spectra += 1;
        scan.deconvoluted_peaks = Some(deconvoluted_peaks);
        scan.precursor_mut().and_then(|prec| {
            let target_mz = prec.mz();
            let _ = precursor_mz
                .iter()
                .find_position(|t| ((**t) - target_mz).abs() < 1e-6)
                .and_then(|(i, _)| {
                    if let Some(peak) = &targets[i] {
                        let orig_charge = prec.ion.charge;
                        let update_ion = if let Some(orig_z) = orig_charge {
                            let t = orig_z == peak.charge;
                            if !t {
                                prog.precursor_charge_state_mismatch += 1;
                            }
                            t
                        } else {
                            true
                        };
                        if update_ion {
                            prec.ion.mz = peak.mz();
                            prec.ion.charge = Some(peak.charge);
                            prec.ion.intensity = peak.intensity;
                        } else {
                            prec.ion.params_mut().push(Param::new_key_value(
                                "mzdeisotope:defaulted".to_string(),
                                true.to_string(),
                            ));
                            prog.precursors_defaulted += 1;
                        }
                    }
                    Some(())
                })
                .or_else(|| {
                    prec.ion.params_mut().push(Param::new_key_value(
                        "mzdeisotope:defaulted".to_string(),
                        true.to_string(),
                    ));
                    prec.ion.params_mut().push(Param::new_key_value(
                        "mzdeisotope:orphan".to_string(),
                        true.to_string(),
                    ));
                    prog.precursors_defaulted += 1;
                    None
                });

            Some(())
        });
    });
    (group_idx, group, prog)
}
