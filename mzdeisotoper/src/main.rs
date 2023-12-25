use std::fs;
use std::io;
use std::path;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Instant;

use clap::Parser;
use itertools::Itertools;
use log;
use pretty_env_logger;
use rayon::prelude::*;

use mzdata::spectrum::MultiLayerSpectrum;
use mzdeisotope::charge::ChargeRange;
use mzdeisotope::scorer::{IsotopicFitFilter, IsotopicPatternScorer};

use mzdata::io::PreBufferedStream;
use mzdata::io::{
    mzml::{MzMLReaderType, MzMLWriterType},
    traits::ScanWriter,
};
use mzdata::prelude::*;
use mzdata::spectrum::{utils::Collator, SignalContinuity, SpectrumGroup};
use mzdata::Param;

use mzdeisotope::api::{DeconvolutionEngine, PeaksAndTargets};
use mzdeisotope::isotopic_model::{IsotopicModel, IsotopicModels, IsotopicPatternParams, PROTON};
use mzdeisotope::scorer::{MSDeconvScorer, MaximizingFitFilter, PenalizedMSDeconvScorer, ScoreType};
use mzdeisotope::solution::DeconvolvedSolutionPeak;

use mzpeaks::{CentroidPeak, PeakCollection, Tolerance};

type SpectrumGroupType = SpectrumGroup<CentroidPeak, DeconvolvedSolutionPeak>;
type SpectrumGroupCollator = Collator<SpectrumGroupType>;
type SpectrumType = MultiLayerSpectrum<CentroidPeak, DeconvolvedSolutionPeak>;

struct SignalParams {
    pub ms1_averaging: usize,
    pub mz_range: (f64, f64),
    pub interpolation_dx: f64,
}

struct DeconvolutionParams<'a, S: IsotopicPatternScorer, F: IsotopicFitFilter> {
    pub scorer: S,
    pub isotopic_model: IsotopicModel<'a>,
    pub fit_filter: F,
    pub isotopic_params: IsotopicPatternParams,
    pub charge_range: ChargeRange,
    pub mz_range: (f64, f64),
}

impl<'a, S: IsotopicPatternScorer, F: IsotopicFitFilter> DeconvolutionParams<'a, S, F> {
    fn new(
        scorer: S,
        isotopic_model: IsotopicModel<'a>,
        fit_filter: F,
        isotopic_params: IsotopicPatternParams,
        charge_range: ChargeRange,
        mz_range: (f64, f64),
    ) -> Self {
        Self {
            scorer,
            isotopic_model,
            fit_filter,
            isotopic_params,
            charge_range,
            mz_range,
        }
    }

    fn build_engine(self) -> DeconvolutionEngine<'a, CentroidPeak, S, F> {
        let mut engine = DeconvolutionEngine::new(
            self.isotopic_params,
            self.isotopic_model.into(),
            self.scorer,
            self.fit_filter,
        );
        engine.populate_isotopic_model_cache(
            self.mz_range.0,
            self.mz_range.1,
            self.charge_range.0,
            self.charge_range.1,
        );
        engine
    }
}

fn prepare_procesing<
    R: ScanSource<CentroidPeak, DeconvolvedSolutionPeak, SpectrumType> + Send,
    S: IsotopicPatternScorer + Send + Sync + Clone + 'static,
    F: IsotopicFitFilter + Send + Sync + Clone + 'static,
    SN: IsotopicPatternScorer + Send + Sync + Clone + 'static,
    FN: IsotopicFitFilter + Send + Sync + Clone + 'static,
>(
    mut reader: R,
    ms1_deconv_params: DeconvolutionParams<'static, S, F>,
    msn_deconv_params: DeconvolutionParams<'static, SN, FN>,
    signal_processing_params: SignalParams,
    sender: Sender<(usize, SpectrumGroup<CentroidPeak, DeconvolvedSolutionPeak>)>,
) -> io::Result<()> {
    reader.reset();
    let init_counter = AtomicU16::new(0);
    let started = Instant::now();

    let build_ms1_engine = thread::spawn(move || ms1_deconv_params.build_engine());
    let build_msn_engine = thread::spawn(move || msn_deconv_params.build_engine());

    let ms1_engine = build_ms1_engine.join().unwrap();
    let msn_engine = build_msn_engine.join().unwrap();

    let (n_ms1_peaks, n_msn_peaks) = if signal_processing_params.ms1_averaging > 0 {
        let (grouper, averager, reprofiler) = reader.into_groups().averaging_deferred(
            signal_processing_params.ms1_averaging,
            signal_processing_params.mz_range.0,
            signal_processing_params.mz_range.1 + signal_processing_params.interpolation_dx,
            signal_processing_params.interpolation_dx,
        );
        grouper
            .enumerate()
            .par_bridge()
            .map_init(
                || (averager.clone(), reprofiler.clone()),
                |(averager, reprofiler), (i, g)| {
                    let (mut g, arrays) = g.reprofile_with_average_with(averager, reprofiler);
                    g.precursor_mut().and_then(|p| {
                        p.arrays = Some(arrays.into());
                        p.description_mut().signal_continuity = SignalContinuity::Profile;
                        Some(())
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
                    deconvolution_transform(ms1_engine, msn_engine, group_idx, group)
                },
            )
            .map(|(group_idx, group, n_ms1_peaks_local, n_msn_peaks_local)| {
                match sender.send((group_idx, group)) {
                    Ok(_) => {}
                    Err(e) => {
                        log::warn!("Failed to send group: {}", e);
                    }
                }
                (n_ms1_peaks_local, n_msn_peaks_local)
            })
            .reduce(
                || (0usize, 0usize),
                |state, counts| (state.0 + counts.0, state.1 + counts.1),
            )
    } else {
        let grouper = reader.into_groups().enumerate().par_bridge();
        grouper
            .map_init(
                || {
                    init_counter.fetch_add(1, Ordering::AcqRel);
                    (ms1_engine.clone(), msn_engine.clone())
                },
                |(ms1_engine, msn_engine), (group_idx, group)| {
                    deconvolution_transform(ms1_engine, msn_engine, group_idx, group)
                },
            )
            .map(|(group_idx, group, n_ms1_peaks_local, n_msn_peaks_local)| {
                match sender.send((group_idx, group)) {
                    Ok(_) => {}
                    Err(e) => {
                        log::warn!("Failed to send group: {}", e);
                    }
                }
                (n_ms1_peaks_local, n_msn_peaks_local)
            })
            .reduce(
                || (0usize, 0usize),
                |state, counts| (state.0 + counts.0, state.1 + counts.1),
            )
    };

    let finished = Instant::now();
    let elapsed = finished - started;
    log::debug!(
        "{} threads run for deconvolution",
        init_counter.load(Ordering::SeqCst)
    );
    log::info!("MS1 Peaks: {n_ms1_peaks}\tMSn Peaks: {n_msn_peaks}");
    log::info!("Elapsed Time: {:0.3?}", elapsed);
    Ok(())
}

fn deconvolution_transform<
    S: IsotopicPatternScorer + Send + 'static,
    F: IsotopicFitFilter + Send + 'static,
    SN: IsotopicPatternScorer + Send + 'static,
    FN: IsotopicFitFilter + Send + 'static,
>(
    ms1_engine: &mut DeconvolutionEngine<'_, CentroidPeak, S, F>,
    msn_engine: &mut DeconvolutionEngine<'_, CentroidPeak, SN, FN>,
    group_idx: usize,
    mut group: SpectrumGroupType,
) -> (
    usize,
    SpectrumGroup<CentroidPeak, DeconvolvedSolutionPeak>,
    usize,
    usize,
) {
    let had_precursor = group.precursor.is_some();
    let mut n_ms1_peaks = 0usize;
    let mut n_msn_peaks = 0usize;

    let precursor_mz: Vec<_> = group
        .products()
        .into_iter()
        .flat_map(|s| s.precursor().and_then(|prec| Some(prec.ion().mz)))
        .collect();
    let targets = match group.precursor_mut() {
        Some(scan) => {
            // log::info!("Processing {} MS{} ({:0.3})", scan.id(), scan.ms_level(), scan.acquisition().start_time());
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

            #[cfg(feature = "verbose")]
            ms1_engine.set_log_file(Some(logfile));

            let PeaksAndTargets {
                deconvoluted_peaks,
                targets,
            } = ms1_engine.deconvolute_peaks_with_targets(
                peaks.clone(),
                Tolerance::PPM(20.0),
                (1, 8),
                1,
                &precursor_mz,
            );
            n_ms1_peaks = deconvoluted_peaks.len();
            scan.deconvoluted_peaks = Some(deconvoluted_peaks);
            targets
        }
        None => precursor_mz.iter().map(|_| None).collect(),
    };

    group.products_mut().iter_mut().for_each(|scan| {
        if !had_precursor {
            log::info!(
                "Processing {} MS{} ({:0.3})",
                scan.id(),
                scan.ms_level(),
                scan.acquisition().start_time()
            );
        }

        #[cfg(feature = "verbose")]
        let logfile = fs::File::create(format!("scan-ms2-{}-log.txt", scan.index())).unwrap();

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

        #[cfg(feature = "verbose")]
        msn_engine.set_log_file(Some(logfile));

        let deconvoluted_peaks =
            msn_engine.deconvolute_peaks(peaks.clone(), Tolerance::PPM(20.0), (1, 8), 1);
        n_msn_peaks += deconvoluted_peaks.len();
        scan.deconvoluted_peaks = Some(deconvoluted_peaks);
        scan.precursor_mut().and_then(|prec| {
            let target_mz = prec.mz();
            let _ = precursor_mz
                .iter()
                .find_position(|t| ((**t) - target_mz).abs() < 1e-6)
                .and_then(|(i, _)| {
                    if let Some(peak) = &targets[i] {
                        // let orig_mz = prec.ion.mz;
                        let orig_charge = prec.ion.charge;
                        let update_ion = if let Some(orig_z) = orig_charge {
                            orig_z == peak.charge
                        } else {
                            true
                        };
                        if update_ion {
                            prec.ion.mz = peak.mz();
                            prec.ion.charge = Some(peak.charge);
                            prec.ion.intensity = peak.intensity;
                        } else {
                            // log::warn!("Expected ion of charge state {} @ {orig_mz:0.3}, found {} @ {:0.3}", orig_charge.unwrap(), peak.charge, peak.mz());
                            prec.ion.params_mut().push(Param::new_key_value(
                                "mzdeisotope:defaulted".to_string(),
                                true.to_string(),
                            ));
                        }
                    };
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
                    None
                });

            Some(prec)
        });
    });
    (group_idx, group, n_ms1_peaks, n_msn_peaks)
}

fn collate_results(
    receiver: Receiver<(usize, SpectrumGroupType)>,
    sender: Sender<(usize, SpectrumGroupType)>,
) {
    let mut collator = SpectrumGroupCollator::default();
    loop {
        match receiver.try_recv() {
            Ok((group_idx, group)) => {
                collator.receive(group_idx, group);
                collator.receive_from(&receiver, 100);
            }
            Err(e) => match e {
                TryRecvError::Empty => {}
                TryRecvError::Disconnected => {
                    collator.done = true;
                    break;
                }
            },
        }

        while let Some((group_idx, group)) = collator.try_next() {
            match sender.send((group_idx, group)) {
                Ok(()) => {}
                Err(e) => {
                    log::error!("Failed to send {group_idx} for writing: {e}")
                }
            }
        }
    }
}

fn write_output<W: io::Write>(
    mut writer: MzMLWriterType<W, CentroidPeak, DeconvolvedSolutionPeak>,
    receiver: Receiver<(usize, SpectrumGroupType)>,
) -> io::Result<()> {
    let mut checkpoint = 0usize;
    let mut time_checkpoint = 0.0;
    let mut scan_counter = 0usize;
    while let Ok((group_idx, group)) = receiver.recv() {
        let scan = group
            .precursor()
            .or_else(|| {
                group
                    .products()
                    .into_iter()
                    .min_by(|a, b| a.start_time().total_cmp(&b.start_time()))
            })
            .unwrap();
        scan_counter += group.total_spectra();
        let scan_time = scan.start_time();
        if ((group_idx - checkpoint) % 100 == 0 && group_idx != 0)
            || (scan_time - time_checkpoint) > 1.0
        {
            log::info!("Completed Group {group_idx} | Scans={scan_counter} Time={scan_time:0.3}");
            checkpoint = group_idx;
            time_checkpoint = scan_time;
        }
        writer.write_group(&group)?;
    }
    match writer.close() {
        Ok(_) => {}
        Err(e) => match e {
            mzdata::MzMLWriterError::IOError(o) => return Err(o),
            _ => Err(io::Error::new(io::ErrorKind::InvalidInput, e))?,
        },
    };
    Ok(())
}

fn make_default_ms1_deconvolution_params(
) -> DeconvolutionParams<'static, PenalizedMSDeconvScorer, MaximizingFitFilter> {
    DeconvolutionParams::new(
        PenalizedMSDeconvScorer::new(0.02, 2.0),
        IsotopicModels::Peptide.into(),
        MaximizingFitFilter::new(10.0),
        Default::default(),
        (1, 8),
        (80.0, 2200.0),
    )
}

fn make_default_msn_deconvolution_params(
) -> DeconvolutionParams<'static, MSDeconvScorer, MaximizingFitFilter> {
    DeconvolutionParams::new(
        MSDeconvScorer::default(),
        IsotopicModels::Peptide.into(),
        MaximizingFitFilter::new(2.0),
        IsotopicPatternParams::new(0.8, 0.001, None, PROTON),
        (1, 8),
        (80.0, 2200.0),
    )
}

fn make_default_signal_processing_params() -> SignalParams {
    SignalParams {
        ms1_averaging: 1,
        mz_range: (80.0, 2200.0),
        interpolation_dx: 0.005,
    }
}

#[derive(Parser, Debug)]
struct MZDeiosotoperArgs {
    pub input_file: String,

    #[arg(short = 'o', long = "output-file", default_value = "-")]
    pub output_file: String,

    #[arg(short = 'g', long = "ms1-averaging-range", default_value_t = 0)]
    pub ms1_averaging_range: usize,

    #[arg(short = 'a', long = "ms1-isotopic-model", default_value = "peptide")]
    pub ms1_isotopic_model: String,

    #[arg(short = 's', long = "ms1-score-thresold", default_value_t = 10.0)]
    pub ms1_score_threshold: ScoreType,

    #[arg(long = "msn-isotopic-model", default_value = "peptide")]
    pub msn_isotopic_model: String,

    #[arg(long = "msn-score-thresold", default_value_t = 2.0)]
    pub msn_score_threshold: ScoreType,
}

fn main_with_reader<
    R: ScanSource<CentroidPeak, DeconvolvedSolutionPeak, SpectrumType>
        + MSDataFileMetadata
        + Send
        + 'static,
>(
    args: MZDeiosotoperArgs,
    reader: R,
) -> io::Result<()> {
    if args.output_file == "-" {
        let outfile = io::stdout();
        let mut writer = MzMLWriterType::<_, CentroidPeak, DeconvolvedSolutionPeak>::new(outfile);
        writer.copy_metadata_from(&reader);
        main_task(args, reader, writer)?;
    } else {
        let mut writer = MzMLWriterType::<_, CentroidPeak, DeconvolvedSolutionPeak>::new(
            io::BufWriter::new(fs::File::create(args.output_file.clone())?),
        );
        writer.copy_metadata_from(&reader);
        main_task(args, reader, writer)?;
    }
    Ok(())
}

fn main_task<
    R: ScanSource<CentroidPeak, DeconvolvedSolutionPeak, SpectrumType>
        + MSDataFileMetadata
        + Send
        + 'static,
    W: io::Write + Send + 'static,
>(
    args: MZDeiosotoperArgs,
    reader: R,
    writer: MzMLWriterType<W, CentroidPeak, DeconvolvedSolutionPeak>,
) -> io::Result<()> {
    let (send_solved, recv_solved) = channel();
    let (send_collated, recv_collated) = channel();

    let mut ms1_args = make_default_ms1_deconvolution_params();
    let mut signal_params = make_default_signal_processing_params();
    let msn_args = make_default_msn_deconvolution_params();

    ms1_args.fit_filter.threshold = args.ms1_score_threshold;
    signal_params.ms1_averaging = args.ms1_averaging_range;

    let read_task = thread::spawn(move || {
        prepare_procesing(reader, ms1_args, msn_args, signal_params, send_solved)
    });

    let collate_task = thread::spawn(move || collate_results(recv_solved, send_collated));

    let write_task = thread::spawn(move || write_output(writer, recv_collated));

    match read_task.join() {
        Ok(o) => o?,
        Err(e) => {
            log::warn!("Failed to join reader task: {e:?}");
        }
    }

    match collate_task.join() {
        Ok(_) => {}
        Err(e) => {
            log::warn!("Failed to join collator task: {e:?}")
        }
    }

    match write_task.join() {
        Ok(o) => o?,
        Err(e) => {
            log::warn!("Failed to join writer task: {e:?}");
        }
    }
    Ok(())
}

fn main() -> io::Result<()> {
    pretty_env_logger::init_timed();

    let args = MZDeiosotoperArgs::parse();
    if args.input_file == "-" {
        let buffered = PreBufferedStream::new(io::stdin())?;
        let reader = MzMLReaderType::<_, CentroidPeak, DeconvolvedSolutionPeak>::new(buffered);
        main_with_reader(args, reader)?;
    } else {
        let reader = MzMLReaderType::<_, CentroidPeak, DeconvolvedSolutionPeak>::open_path(
            path::PathBuf::from(args.input_file.clone()),
        )?;
        main_with_reader(args, reader)?;
    }
    Ok(())
}
