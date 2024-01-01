use std::fs;
use std::io;
use std::path;
use std::str::FromStr;
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
use mzdeisotope::scorer::{IsotopicFitFilter, IsotopicPatternScorer};

use mzdata::io::PreBufferedStream;
use mzdata::io::{
    mzml::{MzMLReaderType, MzMLWriterType},
    ScanWriter,
};
use mzdata::prelude::*;
use mzdata::spectrum::{utils::Collator, SignalContinuity, SpectrumGroup};
use mzdata::Param;

use mzdeisotope::api::{DeconvolutionEngine, PeaksAndTargets};
use mzdeisotope::scorer::ScoreType;
use mzdeisotope::solution::DeconvolvedSolutionPeak;

use mzpeaks::{CentroidPeak, PeakCollection, Tolerance};

mod args;
mod progress;
mod time_range;

use crate::args::{
    make_default_ms1_deconvolution_params, make_default_msn_deconvolution_params,
    make_default_signal_processing_params, ArgIsotopicModels, DeconvolutionBuilderParams,
    DeconvolutionParams, SignalParams,
};
use crate::progress::ProgressRecord;
use crate::time_range::TimeRange;

type SpectrumGroupType = SpectrumGroup<CentroidPeak, DeconvolvedSolutionPeak>;
type SpectrumGroupCollator = Collator<SpectrumGroupType>;
type SpectrumType = MultiLayerSpectrum<CentroidPeak, DeconvolvedSolutionPeak>;

fn prepare_procesing<
    R: RandomAccessSpectrumIterator<CentroidPeak, DeconvolvedSolutionPeak, SpectrumType> + Send,
    S: IsotopicPatternScorer + Send + Sync + Clone + 'static,
    F: IsotopicFitFilter + Send + Sync + Clone + 'static,
    SN: IsotopicPatternScorer + Send + Sync + Clone + 'static,
    FN: IsotopicFitFilter + Send + Sync + Clone + 'static,
>(
    reader: R,
    ms1_deconv_params: DeconvolutionBuilderParams<'static, S, F>,
    msn_deconv_params: DeconvolutionBuilderParams<'static, SN, FN>,
    signal_processing_params: SignalParams,
    sender: Sender<(usize, SpectrumGroup<CentroidPeak, DeconvolvedSolutionPeak>)>,
    time_range: Option<TimeRange>,
) -> io::Result<()> {
    let init_counter = AtomicU16::new(0);
    let started = Instant::now();

    let build_ms1_engine = thread::spawn(move || {
        (
            ms1_deconv_params.make_params(),
            ms1_deconv_params.build_engine(),
        )
    });
    let build_msn_engine = thread::spawn(move || {
        (
            msn_deconv_params.make_params(),
            msn_deconv_params.build_engine(),
        )
    });

    let (ms1_deconv_params, ms1_engine) = build_ms1_engine.join().unwrap();
    let (msn_deconv_params, msn_engine) = build_msn_engine.join().unwrap();

    let mut group_iter = reader.into_groups();
    let end_time = if let Some(time_range) = time_range {
        log::info!("Starting from {}", time_range.start);
        group_iter.start_from_time(time_range.start).unwrap();
        time_range.end
    } else {
        f64::INFINITY
    };

    let prog = if signal_processing_params.ms1_averaging > 0 {
        let (grouper, averager, reprofiler) = group_iter.averaging_deferred(
            signal_processing_params.ms1_averaging,
            signal_processing_params.mz_range.0,
            signal_processing_params.mz_range.1 + signal_processing_params.interpolation_dx,
            signal_processing_params.interpolation_dx,
        );
        grouper
            .enumerate()
            .take_while(|(_, g)| !g.group.iter().any(|s| s.start_time() > end_time))
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
                    deconvolution_transform(
                        ms1_engine,
                        msn_engine,
                        &ms1_deconv_params,
                        &msn_deconv_params,
                        &signal_processing_params,
                        group_idx,
                        group,
                    )
                },
            )
            .map(|(group_idx, group, prog)| {
                match sender.send((group_idx, group)) {
                    Ok(_) => {}
                    Err(e) => {
                        log::warn!("Failed to send group: {}", e);
                    }
                }
                prog
            })
            .reduce(
                || ProgressRecord::default(),
                |mut state, counts| {
                    state += counts;
                    state
                },
            )
    } else {
        let grouper = group_iter
            .enumerate()
            .take_while(|(_, g)| !g.iter().any(|s| s.start_time() > end_time))
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
                    )
                },
            )
            .map(|(group_idx, group, prog)| {
                match sender.send((group_idx, group)) {
                    Ok(_) => {}
                    Err(e) => {
                        log::warn!("Failed to send group: {}", e);
                    }
                }
                prog
            })
            .reduce(
                || ProgressRecord::default(),
                |mut state, counts| {
                    state += counts;
                    state
                },
            )
    };

    let finished = Instant::now();
    let elapsed = finished - started;
    log::debug!(
        "{} threads run for deconvolution",
        init_counter.load(Ordering::SeqCst)
    );
    log::info!("MS1 Spectra: {}", prog.ms1_spectra);
    log::info!("MSn Spectra: {}", prog.msn_spectra);
    log::info!(
        "Precursors Defaulted: {} | Mismatched Charge State: {}",
        prog.precursors_defaulted,
        prog.precursor_charge_state_mismatch
    );
    log::info!("MS1 Peaks: {}", prog.ms1_peaks);
    log::info!("MSn Peaks: {}", prog.msn_peaks);
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
    ms1_deconv_params: &DeconvolutionParams,
    msn_deconv_params: &DeconvolutionParams,
    signal_processing_params: &SignalParams,
    group_idx: usize,
    mut group: SpectrumGroupType,
) -> (
    usize,
    SpectrumGroup<CentroidPeak, DeconvolvedSolutionPeak>,
    ProgressRecord,
) {
    let had_precursor = group.precursor.is_some();
    let mut prog = ProgressRecord::default();

    let precursor_mz: Vec<_> = group
        .products()
        .into_iter()
        .flat_map(|s| s.precursor().and_then(|prec| Some(prec.ion().mz)))
        .collect();
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
                SignalContinuity::Centroid => scan.try_build_centroids().unwrap(),
                SignalContinuity::Profile => {
                    if signal_processing_params.ms1_denoising > 0.0 {
                        log::trace!("Denoising {}", scan.id());
                        if let Err(e) = scan.denoise(signal_processing_params.ms1_denoising) {
                            log::error!("An error occurred while denoising {}: {e}", scan.id());
                        }
                    }
                    scan.pick_peaks(1.0, Default::default()).unwrap();
                    scan.description_mut().signal_continuity = SignalContinuity::Centroid;
                    scan.peaks.as_ref().unwrap()
                }
            };

            let PeaksAndTargets {
                deconvoluted_peaks,
                targets,
            } = ms1_engine
                .deconvolute_peaks_with_targets(
                    peaks.clone(),
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

#[derive(Parser, Debug)]
struct MZDeiosotoperArgs {
    #[arg(help = "The path to read the input spectra from, or if '-' is passed, read from STDIN")]
    pub input_file: String,

    #[arg(
        short = 'o',
        long = "output-file",
        default_value = "-",
        help = "The path to write the output file to, or if '-' is passed, write to STDOUT"
    )]
    pub output_file: String,

    #[arg(short='t', long="threads", default_value_t=-1, help="The number of threads to use, passing a value < 1 to use all available threads")]
    pub threads: i32,

    #[arg(short='r', long="time-range", value_parser=TimeRange::from_str)]
    pub rt_range: Option<TimeRange>,

    #[arg(
        short = 'g',
        long = "ms1-averaging-range",
        default_value_t = 0,
        help = "The number of MS1 spectra before and after to average with prior to peak picking"
    )]
    pub ms1_averaging_range: usize,

    #[arg(
        short = 'b',
        long = "ms1-background-reduction",
        default_value_t = 0.0,
        help = "The magnitude of background noise reduction to use on MS1 spectra prior to peak picking"
    )]
    pub ms1_denoising: f32,

    #[arg(
        short = 'a',
        long = "ms1-isotopic-model",
        default_value = "peptide",
        help = "The isotopic model to use for MS1 spectra"
    )]
    pub ms1_isotopic_model: ArgIsotopicModels,

    #[arg(
        short = 's',
        long = "ms1-score-threshold",
        default_value_t = 10.0,
        help = "The minimum isotopic pattern fit score for MS1 spectra"
    )]
    pub ms1_score_threshold: ScoreType,

    #[arg(
        short = 'A',
        long = "msn-isotopic-model",
        default_value = "peptide",
        help = "The isotopic model to use for MSn spectra"
    )]
    pub msn_isotopic_model: ArgIsotopicModels,

    #[arg(
        short = 'S',
        long = "msn-score-threshold",
        default_value_t = 2.0,
        help = "The minimum isotopic pattern fit score for MSn spectra"
    )]
    pub msn_score_threshold: ScoreType,
}

impl MZDeiosotoperArgs {
    pub fn main(&self) -> io::Result<()> {
        if self.threads > 0 {
            log::debug!("Using {} threads", self.threads);
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.threads as usize)
                .build_global()
                .unwrap();
        }

        self.reader_then()
    }

    fn reader_then(&self) -> io::Result<()> {
        if self.input_file == "-" {
            let buffered = PreBufferedStream::new_with_buffer_size(io::stdin(), 2usize.pow(20))?;
            let reader = MzMLReaderType::<_, CentroidPeak, DeconvolvedSolutionPeak>::new(buffered);
            let spectrum_count_hint = reader.spectrum_count_hint();
            self.writer_then(reader, spectrum_count_hint)?;
        } else {
            let reader = MzMLReaderType::<_, CentroidPeak, DeconvolvedSolutionPeak>::open_path(
                path::PathBuf::from(self.input_file.clone()),
            )?;
            let spectrum_count = Some(reader.len() as u64);
            self.writer_then(reader, spectrum_count)?;
        }
        Ok(())
    }

    fn writer_then<
        R: RandomAccessSpectrumIterator<CentroidPeak, DeconvolvedSolutionPeak, SpectrumType>
            + MSDataFileMetadata
            + Send
            + 'static,
    >(
        &self,
        reader: R,
        spectrum_count: Option<u64>,
    ) -> io::Result<()> {
        if self.output_file == "-" {
            let outfile = io::stdout();
            let mut writer =
                MzMLWriterType::<_, CentroidPeak, DeconvolvedSolutionPeak>::new(outfile);
            writer.copy_metadata_from(&reader);
            if let Some(spectrum_count) = spectrum_count {
                writer.set_spectrum_count(spectrum_count);
            }
            self.run_workflow(reader, writer)?;
        } else {
            let mut writer = MzMLWriterType::<_, CentroidPeak, DeconvolvedSolutionPeak>::new(
                io::BufWriter::new(fs::File::create(self.output_file.clone())?),
            );
            writer.copy_metadata_from(&reader);
            if let Some(spectrum_count) = spectrum_count {
                writer.set_spectrum_count(spectrum_count);
            }
            self.run_workflow(reader, writer)?;
        }
        Ok(())
    }

    fn run_workflow<
        R: RandomAccessSpectrumIterator<CentroidPeak, DeconvolvedSolutionPeak, SpectrumType>
            + MSDataFileMetadata
            + Send
            + 'static,
        W: io::Write + Send + 'static,
    >(
        &self,
        reader: R,
        writer: MzMLWriterType<W, CentroidPeak, DeconvolvedSolutionPeak>,
    ) -> io::Result<()> {
        let (send_solved, recv_solved) = channel();
        let (send_collated, recv_collated) = channel();

        let mut ms1_args = make_default_ms1_deconvolution_params();
        let mut signal_params = make_default_signal_processing_params();
        let mut msn_args = make_default_msn_deconvolution_params();

        ms1_args.fit_filter.threshold = self.ms1_score_threshold;
        ms1_args.isotopic_model = self.ms1_isotopic_model.into();

        msn_args.fit_filter.threshold = self.msn_score_threshold;
        msn_args.isotopic_model = self.msn_isotopic_model.into();

        signal_params.ms1_averaging = self.ms1_averaging_range;
        signal_params.ms1_denoising = self.ms1_denoising;

        let rt_range = self.rt_range;

        let read_task = thread::spawn(move || {
            prepare_procesing(
                reader,
                ms1_args,
                msn_args,
                signal_params,
                send_solved,
                rt_range,
            )
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
}

fn main() -> io::Result<()> {
    pretty_env_logger::init_timed();

    let args = MZDeiosotoperArgs::parse();
    args.main()?;
    Ok(())
}
