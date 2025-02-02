use std::io;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;
use std::time::Instant;

use crossbeam_channel::Sender;

use mzdata::io::MassSpectrometryFormat;
use rayon::prelude::*;

use tracing::{debug, info, trace, warn};

#[cfg(feature = "mzmlb")]
use mzdata::io::mzmlb::{MzMLbReaderType, MzMLbWriterBuilder};
use mzdata::prelude::*;
use mzdata::spectrum::SignalContinuity;

use mzpeaks::Tolerance;

use mzdeisotope::scorer::{IsotopicFitFilter, IsotopicPatternScorer};

use crate::args::{DeconvolutionBuilderParams, PrecursorProcessing, SignalParams};
use crate::deconv::deconvolution_transform;
use crate::deconv_im::deconvolution_transform_im;
use crate::progress::ProgressRecord;
use crate::selection_targets::{FrameMSnTargetTracking, MSnTargetTracking, SpectrumGroupTiming, TargetTrackingFrameGroup};
use crate::time_range::TimeRange;
use crate::types::{
    CFeature, CPeak, DFeature, DPeak, FrameGroupType, FrameType, SpectrumGroupType, SpectrumType,
};
use crate::FeatureExtractionParams;

pub fn prepare_procesing<
    R: RandomAccessSpectrumIterator<CPeak, DPeak, SpectrumType> + Send + MSDataFileMetadata,
    S: IsotopicPatternScorer + Send + Sync + Clone + 'static,
    F: IsotopicFitFilter + Send + Sync + Clone + 'static,
    SN: IsotopicPatternScorer + Send + Sync + Clone + 'static,
    FN: IsotopicFitFilter + Send + Sync + Clone + 'static,
>(
    reader: R,
    ms1_deconv_params: DeconvolutionBuilderParams<'static, S, F>,
    msn_deconv_params: DeconvolutionBuilderParams<'static, SN, FN>,
    signal_processing_params: SignalParams,
    sender: Sender<(usize, SpectrumGroupType)>,
    time_range: Option<TimeRange>,
    precursor_processing: Option<PrecursorProcessing>,
    writer_format: MassSpectrometryFormat,
) -> io::Result<ProgressRecord> {
    let init_counter = AtomicU32::new(0);
    let init_averager_counter = AtomicU32::new(0);
    let started = Instant::now();
    let precursor_processing = precursor_processing.unwrap_or_default();

    let build_ms1_engine = thread::spawn(move || ms1_deconv_params.make_params_and_engine());
    let build_msn_engine = thread::spawn(move || msn_deconv_params.make_params_and_engine());

    let mut group_iter = reader.into_groups();
    let end_time = if let Some(time_range) = time_range {
        info!("Starting from {}", time_range.start);
        group_iter.start_from_time(time_range.start).unwrap();
        time_range.end
    } else {
        f64::INFINITY
    };

    let (ms1_deconv_params, ms1_engine) = build_ms1_engine.join().unwrap();
    let (msn_deconv_params, msn_engine) = build_msn_engine.join().unwrap();

    let prog: ProgressRecord = if signal_processing_params.ms1_averaging > 0 {
        let (grouper, averager, reprofiler) = group_iter
            .track_precursors(2.0, Tolerance::PPM(5.0))
            .averaging_deferred(
                signal_processing_params.ms1_averaging,
                signal_processing_params.mz_range.0,
                signal_processing_params.mz_range.1 + signal_processing_params.interpolation_dx,
                signal_processing_params.interpolation_dx,
            );
        grouper
            .enumerate()
            .take_while(|(_, g)| (g.earliest_time().unwrap_or_default() <= end_time))
            .par_bridge()
            .map_init(
                || {
                    init_averager_counter.fetch_add(1, Ordering::AcqRel);
                    (averager.clone(), reprofiler.clone())
                },
                |(averager, reprofiler), (i, g)| {
                    let span = tracing::debug_span!(
                        "averaging precursor",
                        scan_id = g.precursor().map(|s| s.id()),
                        group_idx = i
                    );
                    let _entered = span.enter();
                    let (mut g, arrays) = g.reprofile_with_average_with(averager, reprofiler);
                    if let Some(p) = g.precursor_mut() {
                        if tracing::enabled!(tracing::Level::TRACE) {
                            tracing::trace!(
                                "Averaging precursor spectrum {} {} @ {}",
                                p.id(),
                                p.index(),
                                p.start_time()
                            )
                        }
                        p.arrays = Some(arrays.into());
                        p.description_mut().signal_continuity = SignalContinuity::Profile;
                    }
                    (i, g)
                },
            )
            .map_init(
                || {
                    init_counter.fetch_add(1, Ordering::AcqRel);
                    (ms1_engine.clone(), msn_engine.clone())
                },
                |(ms1_engine, msn_engine), (group_idx, group)| {
                    trace!("Processing group {group_idx}");
                    deconvolution_transform(
                        ms1_engine,
                        msn_engine,
                        &ms1_deconv_params,
                        &msn_deconv_params,
                        &signal_processing_params,
                        group_idx,
                        group,
                        precursor_processing,
                        writer_format,
                    )
                },
            )
            .map(|(group_idx, group, prog)| {
                if tracing::event_enabled!(tracing::Level::TRACE) {
                    let tid = thread::current().id();
                    let v: Vec<_> = group.iter().map(|s| s.index()).collect();
                    trace!("{tid:?}: Sending group {group_idx} containing {:?}", v);
                }
                if let Err(e) = sender.send((group_idx, group)) {
                    warn!("Failed to send group: {}", e);
                }
                prog
            })
            .fold(ProgressRecord::default, ProgressRecord::sum)
            .sum()
    } else {
        let grouper = group_iter
            .track_precursors(2.0, Tolerance::PPM(5.0))
            .enumerate()
            .take_while(|(_, g)| (g.earliest_time().unwrap_or_default() <= end_time))
            .par_bridge();
        grouper
            .map_init(
                || {
                    init_counter.fetch_add(1, Ordering::AcqRel);
                    (ms1_engine.clone(), msn_engine.clone())
                },
                |(ms1_engine, msn_engine), (group_idx, group)| {
                    tracing::trace!("Processing group {group_idx}");
                    deconvolution_transform(
                        ms1_engine,
                        msn_engine,
                        &ms1_deconv_params,
                        &msn_deconv_params,
                        &signal_processing_params,
                        group_idx,
                        group,
                        precursor_processing,
                        writer_format,
                    )
                },
            )
            .map(|(group_idx, group, prog)| {
                if tracing::event_enabled!(tracing::Level::TRACE) {
                    let tid = thread::current().id();
                    let v: Vec<_> = group.iter().map(|s| s.index()).collect();
                    trace!("{tid:?}: Sending group {group_idx} containing {:?}", v);
                }
                if let Err(e) = sender.send((group_idx, group)) {
                    warn!("Failed to send group: {}", e);
                }
                prog
            })
            .fold(ProgressRecord::default, ProgressRecord::sum)
            .sum()
    };

    let finished = Instant::now();
    let elapsed = finished - started;
    debug!(
        "{} workers run for deconvolution",
        init_counter.load(Ordering::SeqCst)
    );
    debug!(
        "{} workers run for averaging",
        init_averager_counter.load(Ordering::SeqCst)
    );
    let spectra_per_second =
        (prog.ms1_spectra + prog.msn_spectra) as f64 / elapsed.as_secs() as f64;
    info!(
        "Elapsed Time: {:0.3?} ({:0.2} spectra/sec)",
        elapsed, spectra_per_second
    );
    Ok(prog)
}

pub fn prepare_procesing_im<
    R: RandomAccessIonMobilityFrameIterator<CFeature, DFeature, FrameType> + Send,
    S: IsotopicPatternScorer + Send + Sync + Clone + 'static,
    F: IsotopicFitFilter + Send + Sync + Clone + 'static,
    SN: IsotopicPatternScorer + Send + Sync + Clone + 'static,
    FN: IsotopicFitFilter + Send + Sync + Clone + 'static,
>(
    reader: R,
    ms1_deconv_params: DeconvolutionBuilderParams<'static, S, F>,
    msn_deconv_params: DeconvolutionBuilderParams<'static, SN, FN>,
    extraction_params: FeatureExtractionParams,
    msn_extraction_params: FeatureExtractionParams,
    sender: Sender<(usize, TargetTrackingFrameGroup<CFeature, DFeature, FrameGroupType>)>,
    time_range: Option<TimeRange>,
    precursor_processing: Option<PrecursorProcessing>,
    writer_format: MassSpectrometryFormat,
) -> io::Result<ProgressRecord> {
    let init_counter = AtomicU32::new(0);
    let init_averager_counter = AtomicU32::new(0);

    let started = Instant::now();
    let precursor_processing = precursor_processing.unwrap_or_default();

    let build_ms1_engine = thread::spawn(move || {
        (
            ms1_deconv_params.make_params(),
            ms1_deconv_params.build_feature_engine(),
        )
    });
    let build_msn_engine = thread::spawn(move || {
        (
            msn_deconv_params.make_params(),
            msn_deconv_params.build_feature_engine(),
        )
    });

    let mut group_iter = reader.into_groups();
    let end_time = if let Some(time_range) = time_range {
        info!("Starting from {}", time_range.start);
        group_iter.start_from_time(time_range.start).unwrap();
        time_range.end
    } else {
        f64::INFINITY
    };

    let (ms1_deconv_params, ms1_engine) = build_ms1_engine.join().unwrap();
    let (msn_deconv_params, msn_engine) = build_msn_engine.join().unwrap();

    let mut grouper = group_iter
        .track_precursors(2.0, Tolerance::PPM(5.0))
        .enumerate()
        .take_while(|(_, g)| {
            g.earliest_frame()
                .map(|f| f.start_time())
                .unwrap_or_default()
                <= end_time
        })
        .par_bridge();

    let prog: ProgressRecord = grouper
        .map_init(
            || {
                init_counter.fetch_add(1, Ordering::AcqRel);
                (ms1_engine.clone(), msn_engine.clone())
            },
            |(ms1_engine, msn_engine), (group_idx, group)| {
                tracing::trace!("Processing group {group_idx}");
                deconvolution_transform_im(
                    ms1_engine,
                    msn_engine,
                    &ms1_deconv_params,
                    &msn_deconv_params,
                    &extraction_params,
                    &msn_extraction_params,
                    group_idx,
                    group,
                    precursor_processing,
                    writer_format,
                )
            },
        )
        .map(|(group_idx, group, prog)| {
            if tracing::event_enabled!(tracing::Level::TRACE) {
                let tid = thread::current().id();
                let v: Vec<_> = group.iter().map(|s| s.index()).collect();
                trace!("{tid:?}: Sending group {group_idx} containing {:?}", v);
            }
            if let Err(e) = sender.send((group_idx, group)) {
                warn!("Failed to send group: {}", e);
            }
            prog
        })
        .fold(ProgressRecord::default, ProgressRecord::sum)
        .sum();

    let finished = Instant::now();
    let elapsed = finished - started;
    debug!(
        "{} workers run for deconvolution",
        init_counter.load(Ordering::SeqCst)
    );
    debug!(
        "{} workers run for averaging",
        init_averager_counter.load(Ordering::SeqCst)
    );
    let spectra_per_second =
        (prog.ms1_spectra + prog.msn_spectra) as f64 / elapsed.as_secs() as f64;
    info!(
        "Elapsed Time: {:0.3?} ({:0.2} spectra/sec)",
        elapsed, spectra_per_second
    );
    Ok(prog)
}
