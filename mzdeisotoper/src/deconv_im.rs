use mzdata::{io::MassSpectrometryFormat, prelude::*, spectrum::bindata::ArrayRetrievalError};
use mzpeaks::{
    coordinate::{IntervalTree, SimpleInterval},
    IonMobility,
};

use mzdata::mzsignal::prelude::*;
use mzdeisotope::scorer::{IsotopicFitFilter, IsotopicPatternScorer};
use mzdeisotope_map::FeatureDeconvolutionEngine;
use tracing::debug;

use crate::{
    args::{DeconvolutionParams, PrecursorProcessing},
    progress::ProgressRecord,
    selection_targets::TargetTrackingFrameGroup,
    types::{CFeature, DFeature, FrameGroupType, FrameType},
    write::postprocess_frames,
    FeatureExtractionParams,
};

fn extract_features_ms1(
    frame: &mut FrameType,
    precursor_processing: &PrecursorProcessing,
    extraction_params: &FeatureExtractionParams,
    selected_mz_ranges: &[(f64, f64)],
) -> Result<(), ArrayRetrievalError> {
    match precursor_processing {
        PrecursorProcessing::SelectedPrecursors
        | PrecursorProcessing::Full
        | PrecursorProcessing::MS1Only
        | PrecursorProcessing::DIA => {
            frame.extract_features_simple(
                extraction_params.error_tolerance,
                extraction_params.minimum_size,
                extraction_params.maximum_time_gap,
                None,
            )?;

            if matches!(
                precursor_processing,
                PrecursorProcessing::SelectedPrecursors
            ) {
                let mut features = frame.features.take().unwrap();
                let tree: IntervalTree<f64, SimpleInterval<f64>> = selected_mz_ranges
                    .iter()
                    .map(|(lo, hi)| SimpleInterval::new(*lo, *hi))
                    .collect();
                features = features
                    .into_iter()
                    .filter(|f| tree.contains(&f.mz()))
                    .collect();
                frame.features = Some(features);
            }

            if extraction_params.smoothing > 0 {
                frame.features.as_mut().unwrap().iter_mut().for_each(|f| {
                    f.smooth(extraction_params.smoothing);
                });
            }
        }
        PrecursorProcessing::TandemOnly => {}
    }

    Ok(())
}

fn extract_features_msn(
    frame: &mut FrameType,
    extraction_params: &FeatureExtractionParams,
) -> Result<(), ArrayRetrievalError> {
    frame.extract_features_simple(
        extraction_params.error_tolerance,
        extraction_params.minimum_size,
        extraction_params.maximum_time_gap,
        None,
    )?;
    if extraction_params.smoothing > 0 {
        frame.features.as_mut().unwrap().iter_mut().for_each(|f| {
            f.smooth(extraction_params.smoothing);
        });
    }
    Ok(())
}

#[tracing::instrument(
    level = "debug",
    skip(
        ms1_engine,
        msn_engine,
        ms1_deconv_params,
        msn_deconv_params,
        extraction_params,
        msn_extraction_params,
        group,
        precursor_processing,
        writer_format
    ),
    name = "deconvolution_transform_im"
)]
pub fn deconvolution_transform_im<
    S: IsotopicPatternScorer + Send + 'static,
    F: IsotopicFitFilter + Send + 'static,
    SN: IsotopicPatternScorer + Send + 'static,
    FN: IsotopicFitFilter + Send + 'static,
>(
    ms1_engine: &mut FeatureDeconvolutionEngine<'_, IonMobility, CFeature, S, F>,
    msn_engine: &mut FeatureDeconvolutionEngine<'_, IonMobility, CFeature, SN, FN>,
    ms1_deconv_params: &DeconvolutionParams,
    msn_deconv_params: &DeconvolutionParams,
    extraction_params: &FeatureExtractionParams,
    msn_extraction_params: &FeatureExtractionParams,
    group_idx: usize,
    mut group: TargetTrackingFrameGroup<CFeature, DFeature, FrameGroupType>,
    precursor_processing: PrecursorProcessing,
    #[allow(unused)] writer_format: MassSpectrometryFormat,
) -> (
    usize,
    TargetTrackingFrameGroup<CFeature, DFeature, FrameGroupType>,
    ProgressRecord,
) {
    let had_precursor = group.precursor().is_some();
    let mut prog = ProgressRecord::default();
    let is_dia = matches!(precursor_processing, PrecursorProcessing::DIA);

    let selected_mz_ranges = group.selected_intervals(1.0, 3.0);

    #[allow(unused)]
    let precursor_im_mz: Vec<_> = if is_dia {
        vec![]
    } else {
        group
            .products()
            .iter()
            .flat_map(|s| {
                s.precursor().map(|prec| {
                    let ion = prec.ion();
                    (ion.ion_mobility(), ion.mz)
                })
            })
            .collect()
    };

    match group.precursor_mut() {
        Some(frame) => {
            if tracing::enabled!(tracing::Level::DEBUG) {
                debug!(
                    "Processing {} MS{} ({:0.3})",
                    frame.id(),
                    frame.ms_level(),
                    frame.start_time()
                );
            }

            extract_features_ms1(
                frame,
                &precursor_processing,
                extraction_params,
                &selected_mz_ranges,
            )
            .unwrap();

            let charge_range = match frame.polarity() {
                mzdata::spectrum::ScanPolarity::Unknown
                | mzdata::spectrum::ScanPolarity::Positive => {
                    let (mut low, mut high) = ms1_deconv_params.charge_range;
                    low = low.abs();
                    high = high.abs();
                    (low.min(high), high.max(low))
                }
                mzdata::spectrum::ScanPolarity::Negative => {
                    let (mut low, mut high) = ms1_deconv_params.charge_range;
                    low = low.abs() * -1;
                    high = high.abs() * -1;
                    (low.min(high), high.max(low))
                }
            };

            if let Some(features) = frame.features.clone() {
                debug!(
                    "Deconvolving {} with {} features",
                    frame.id(),
                    features.len()
                );
                let deconv_features = ms1_engine
                    .deconvolute_features(
                        features,
                        extraction_params.error_tolerance,
                        charge_range,
                        extraction_params.minimum_size,
                        extraction_params.maximum_time_gap,
                        5.0,
                        ms1_deconv_params.max_missed_peaks as usize,
                    )
                    .unwrap();
                prog.ms1_peaks += deconv_features.len();
                prog.ms1_spectra += 1;

                frame.deconvoluted_features = Some(deconv_features);
            }
        }
        None => {}
    };

    group
        .products_mut()
        .iter_mut()
        .enumerate()
        .for_each(|(_frame_i, frame)| {
            if !had_precursor && tracing::enabled!(tracing::Level::DEBUG) {
                tracing::debug!(
                    "Processing {} MS{} ({:0.3})",
                    frame.id(),
                    frame.ms_level(),
                    frame.acquisition().start_time()
                );
            }

            let precursor_charge = frame
                .precursor()
                .and_then(|prec| prec.charge())
                .unwrap_or(msn_deconv_params.charge_range.1);

            let mut msn_charge_range = msn_deconv_params.charge_range;

            if precursor_charge.abs() < msn_charge_range.1.abs() && precursor_charge.abs() != 0 {
                msn_charge_range.1 = precursor_charge;
            }

            msn_charge_range = match frame.polarity() {
                mzdata::spectrum::ScanPolarity::Unknown
                | mzdata::spectrum::ScanPolarity::Positive => {
                    let (mut low, mut high) = msn_charge_range;
                    low = low.abs();
                    high = high.abs();
                    (low.min(high), high.max(low))
                }
                mzdata::spectrum::ScanPolarity::Negative => {
                    let (mut low, mut high) = msn_charge_range;
                    low = low.abs() * -1;
                    high = high.abs() * -1;
                    (low.min(high), high.max(low))
                }
            };

            extract_features_msn(frame, msn_extraction_params).unwrap();

            if let Some(features) = frame.features.clone() {
                debug!(
                    "Deconvolving {} with {} features (MSn)",
                    frame.id(),
                    features.len()
                );
                let deconvoluted_features = msn_engine
                    .deconvolute_features(
                        features,
                        msn_extraction_params.error_tolerance,
                        msn_charge_range,
                        msn_extraction_params.minimum_size,
                        msn_extraction_params.maximum_time_gap,
                        5.0,
                        msn_deconv_params.max_missed_peaks as usize,
                    )
                    .unwrap();
                prog.msn_peaks += deconvoluted_features.len();
                prog.msn_spectra += 1;
                frame.deconvoluted_features = Some(deconvoluted_features);
            }
        });

    (_, group) = postprocess_frames(group_idx, group, writer_format);
    (group_idx, group, prog)
}
