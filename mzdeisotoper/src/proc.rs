use std::io;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::mpsc::SyncSender;
use std::thread;
use std::time::Instant;

use rayon::prelude::*;

use tracing::{debug, info, warn};

#[cfg(feature = "mzmlb")]
use mzdata::io::mzmlb::{MzMLbReaderType, MzMLbWriterBuilder};
use mzdata::prelude::*;
use mzdata::spectrum::SignalContinuity;

use mzpeaks::Tolerance;

use mzdeisotope::scorer::{IsotopicFitFilter, IsotopicPatternScorer};

use crate::deconv::deconvolution_transform;
use crate::progress::ProgressRecord;
use crate::time_range::TimeRange;
use crate::selection_targets::{MSnTargetTracking, SpectrumGroupTiming};
use crate::types::{CPeak, DPeak, SpectrumGroupType, SpectrumType};
use crate::args::{
    DeconvolutionBuilderParams, PrecursorProcessing, SignalParams,
};

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
    sender: SyncSender<(usize, SpectrumGroupType)>,
    time_range: Option<TimeRange>,
    precursor_processing: Option<PrecursorProcessing>,
) -> io::Result<ProgressRecord> {
    let init_counter = AtomicU16::new(0);
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
            .take_while(|(_, g)| !(g.earliest_time().unwrap_or_default() > end_time))
            .par_bridge()
            .map_init(
                || (averager.clone(), reprofiler.clone()),
                |(averager, reprofiler), (i, g)| {
                    let (mut g, arrays) = g.reprofile_with_average_with(averager, reprofiler);
                    g.precursor_mut().map(|p| {
                        p.arrays = Some(arrays.into());
                        p.description_mut().signal_continuity = SignalContinuity::Profile;
                        ()
                    });
                    (i, g)
                },
            )
            .map_init(
                || {
                    init_counter.fetch_add(1, Ordering::AcqRel);
                    (ms1_engine.clone(), msn_engine.clone())
                },
                |(ms1_engine, msn_engine), (group_idx, group)| {
                    deconvolution_transform(
                        ms1_engine,
                        msn_engine,
                        &ms1_deconv_params,
                        &msn_deconv_params,
                        &signal_processing_params,
                        group_idx,
                        group,
                        precursor_processing,
                    )
                },
            )
            .map(|(group_idx, group, prog)| {
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
                    deconvolution_transform(
                        ms1_engine,
                        msn_engine,
                        &ms1_deconv_params,
                        &msn_deconv_params,
                        &signal_processing_params,
                        group_idx,
                        group,
                        precursor_processing,
                    )
                },
            )
            .map(|(group_idx, group, prog)| {
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
        "{} threads run for deconvolution",
        init_counter.load(Ordering::SeqCst)
    );
    let spectra_per_second =
        (prog.ms1_spectra + prog.msn_spectra) as f64 / elapsed.as_secs() as f64;
    info!(
        "Elapsed Time: {:0.3?} ({:0.2} spectra/sec)",
        elapsed, spectra_per_second
    );
    Ok(prog)
}
