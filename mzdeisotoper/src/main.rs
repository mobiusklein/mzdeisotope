use std::fs;
use std::io;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::thread;
use std::time::Instant;

use clap::Parser;
use flate2::write::GzEncoder;
use flate2::Compression;
use log;

use pretty_env_logger;
use rayon::prelude::*;
use thiserror::Error;

#[cfg(feature = "mzmlb")]
use mzdata::io::mzmlb::{MzMLbReaderType, MzMLbWriterBuilder};
use mzdata::io::{
    infer_format, infer_from_path, infer_from_stream,
    mgf::{MGFReaderType, MGFWriterType},
    mzml::{MzMLReaderType, MzMLWriterType},
    MassSpectrometryFormat, PreBufferedStream, RestartableGzDecoder, StreamingSpectrumIterator,
};
use mzdata::prelude::*;
use mzdata::spectrum::SignalContinuity;
#[cfg(feature = "mzmlb")]
use std::env;

use mzdeisotope::scorer::ScoreType;
use mzdeisotope::scorer::{IsotopicFitFilter, IsotopicPatternScorer};

use mzpeaks::Tolerance;

mod args;
mod deconv;
mod progress;
mod selection_targets;
mod time_range;
mod types;
mod write;

use crate::args::{
    make_default_ms1_deconvolution_params, make_default_msn_deconvolution_params,
    make_default_signal_processing_params, ArgChargeRange, ArgIsotopicModels,
    DeconvolutionBuilderParams, PrecursorProcessing, SignalParams,
};

#[allow(unused)]
use crate::write::{collate_results, collate_results_spectra, write_output, write_output_spectra};

use crate::deconv::deconvolution_transform;
use crate::progress::ProgressRecord;
use crate::selection_targets::{MSnTargetTracking, SpectrumGroupTiming};
use crate::time_range::TimeRange;
use crate::types::{CPeak, DPeak, SpectrumGroupType, SpectrumType};

fn prepare_procesing<
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
        log::info!("Starting from {}", time_range.start);
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
                        precursor_processing,
                    )
                },
            )
            .map(|(group_idx, group, prog)| {
                if let Err(e) = sender.send((group_idx, group)) {
                    log::warn!("Failed to send group: {}", e);
                }
                prog
            })
            .fold(ProgressRecord::default, ProgressRecord::sum)
            .sum()
    } else {
        let grouper = group_iter
            .track_precursors(2.0, Tolerance::PPM(5.0))
            .enumerate()
            .take_while(|(_, g)| !(g.earliest_time().unwrap_or_default() > end_time))
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
                    log::warn!("Failed to send group: {}", e);
                }
                prog
            })
            .fold(ProgressRecord::default, ProgressRecord::sum)
            .sum()
    };

    let finished = Instant::now();
    let elapsed = finished - started;
    log::debug!(
        "{} threads run for deconvolution",
        init_counter.load(Ordering::SeqCst)
    );
    log::info!("Elapsed Time: {:0.3?}", elapsed);
    Ok(prog)
}

fn non_negative_float_f32(s: &str) -> Result<f32, String> {
    let value = s.parse::<f32>().map_err(|e| e.to_string())?;
    if value < 0.0 {
        return Err(format!("`{s}` is less than zero"));
    } else {
        Ok(value)
    }
}

#[derive(Debug, Error)]
pub enum MZDeisotoperError {
    #[error("An IO error occurred: {0}")]
    IOError(
        #[source]
        #[from]
        io::Error,
    ),
    #[error("The input file format for {0} was either unknown or not supported ({1:?})")]
    FormatUnknownOrNotSupportedError(String, MassSpectrometryFormat),
    #[error("The input file format from STDIN was either unknown or not supported ({0:?})")]
    FormatUnknownOrNotSupportedErrorStdIn(MassSpectrometryFormat),
    #[error("The output file format for {0} was either unknown or not supported ({1:?})")]
    OutputFormatUnknownOrNotSupportedError(String, MassSpectrometryFormat),
    #[error(
        "The input stream was detected to be gzip compressed, which is not currently supported"
    )]
    CompressedInputError(String),
}

/// Deisotoping and charge state deconvolution of mass spectrometry files.
///
/// Read a file or stream, transform the spectra, and write out a processed mzML
/// file or stream.
#[derive(Parser, Debug)]
#[command(author, version)]
pub struct MZDeiosotoperArgs {
    /// The path to read the input spectra from, or if '-' is passed, read from STDIN
    #[arg()]
    pub input_file: String,

    /// The path to write the output file to, or if '-' is passed, write to STDOUT.
    ///
    /// If a path is specified, the output format is inferred, otherwise mzML is assumed.
    #[arg(short = 'o', long = "output-file", default_value = "-")]
    pub output_file: PathBuf,

    /// The number of threads to use, passing a value < 1 to use all available threads
    #[arg(
        short='t',
        long="threads",
        default_value_t=-1,
    )]
    pub threads: i32,

    /// The time range to process, denoted [start?]-[stop?]
    #[arg(
        short='r',
        long="time-range",
        value_parser=TimeRange::from_str,
        value_name="BEGIN-END",
        long_help=r#"The time range to process, denoted [start?]-[stop?]

If a start is not specified, processing begins from the start of the run.
If a stop is not specified, processing stops at the end of the run.
"#
    )]
    pub time_range: Option<TimeRange>,

    /// The number of MS1 spectra before and after to average with prior to peak picking
    #[arg(
        short = 'g',
        long = "ms1-averaging-range",
        default_value_t = 0,
        value_parser = clap::value_parser!(u32).range(0..),
    )]
    pub ms1_averaging_range: u32,

    /// The magnitude of background noise reduction to use on MS1 spectra prior to peak picking
    #[arg(
        short = 'b',
        long = "ms1-background-reduction",
        default_value_t = 0.0,
        value_parser = non_negative_float_f32
    )]
    pub ms1_denoising: f32,

    /// The isotopic model to use for MS1 spectra
    #[arg(short = 'a', long = "ms1-isotopic-model", default_value = "peptide")]
    pub ms1_isotopic_model: ArgIsotopicModels,

    /// The minimum isotopic pattern fit score for MS1 spectra
    #[arg(short = 's', long = "ms1-score-threshold", default_value_t = 20.0)]
    pub ms1_score_threshold: ScoreType,

    /// The isotopic model to use for MSn spectra
    #[arg(short = 'A', long = "msn-isotopic-model", default_value = "peptide")]
    pub msn_isotopic_model: ArgIsotopicModels,

    /// The minimum isotopic pattern fit score for MSn spectra
    #[arg(short = 'S', long = "msn-score-threshold", default_value_t = 10.0)]
    pub msn_score_threshold: ScoreType,

    #[arg(
        short = 'v',
        long = "precursor-processing",
        default_value = "selected-precursors",
        help = "How to treat precursor ranges"
    )]
    pub precursor_processing: PrecursorProcessing,

    /// The range of charge states to consider for each peak denoted [low]-[high] or [high]
    #[arg(
        short = 'z',
        long = "charge-range",
        default_value_t=ArgChargeRange(1, 8),
    )]
    pub charge_range: ArgChargeRange,

    /// The maximum number of missed peaks for MS1 spectra
    #[arg(short = 'm', long = "max-missed-peaks", default_value_t = 1)]
    pub ms1_missed_peaks: u16,

    /// The maximum number of missed peaks for MSn spectra
    #[arg(short = 'M', long = "msn-max-missed-peaks", default_value_t = 1)]
    pub msn_missed_peaks: u16,
}

impl MZDeiosotoperArgs {
    pub fn set_threadpool(&self) {
        if self.threads > 0 {
            log::debug!("Using {} threads", self.threads);
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.threads as usize)
                .build_global()
                .unwrap();
        }
    }

    pub fn main(&self) -> Result<(), MZDeisotoperError> {
        self.set_threadpool();
        self.reader_then()
    }

    fn reader_then(&self) -> Result<(), MZDeisotoperError> {
        if self.input_file == "-" {
            let mut buffered =
                PreBufferedStream::new_with_buffer_size(io::stdin(), 2usize.pow(20))?;
            let (ms_format, compressed) = infer_from_stream(&mut buffered)?;
            log::debug!("Detected {ms_format:?} from STDIN (compressed? {compressed})");
            match ms_format {
                MassSpectrometryFormat::MGF => {
                    if compressed {
                        let reader = StreamingSpectrumIterator::new(MGFReaderType::new(
                            RestartableGzDecoder::new(io::BufReader::new(buffered)),
                        ));
                        let spectrum_count = reader.spectrum_count_hint();
                        self.writer_then(reader, spectrum_count)?;
                    } else {
                        let reader = StreamingSpectrumIterator::new(MGFReaderType::new(buffered));
                        let spectrum_count = reader.spectrum_count_hint();
                        self.writer_then(reader, spectrum_count)?;
                    }
                }
                MassSpectrometryFormat::MzML => {
                    if compressed {
                        let reader = StreamingSpectrumIterator::new(MzMLReaderType::new(
                            RestartableGzDecoder::new(io::BufReader::new(buffered)),
                        ));
                        let spectrum_count = reader.spectrum_count_hint();
                        self.writer_then(reader, spectrum_count)?;
                    } else {
                        let reader = StreamingSpectrumIterator::new(MzMLReaderType::new(buffered));
                        let spectrum_count = reader.spectrum_count_hint();
                        self.writer_then(reader, spectrum_count)?;
                    }
                }
                _ => {
                    return Err(MZDeisotoperError::FormatUnknownOrNotSupportedErrorStdIn(
                        ms_format,
                    ))
                }
            }
        } else {
            let (ms_format, compressed) = infer_format(&self.input_file)?;
            log::debug!("Detected {ms_format:?} from path (compressed? {compressed})");
            match ms_format {
                MassSpectrometryFormat::MGF => {
                    if compressed {
                        let fh = RestartableGzDecoder::new(io::BufReader::new(fs::File::open(
                            &self.input_file,
                        )?));
                        let reader = StreamingSpectrumIterator::new(MGFReaderType::new(fh));
                        let spectrum_count = Some(reader.len() as u64);
                        self.writer_then(reader, spectrum_count)?;
                    } else {
                        let reader = MGFReaderType::open_path(self.input_file.clone())?;
                        let spectrum_count = Some(reader.len() as u64);
                        self.writer_then(reader, spectrum_count)?;
                    }
                }
                MassSpectrometryFormat::MzML => {
                    if compressed {
                        let fh = RestartableGzDecoder::new(io::BufReader::new(fs::File::open(
                            &self.input_file,
                        )?));
                        let reader = StreamingSpectrumIterator::new(MzMLReaderType::new(fh));
                        let spectrum_count = Some(reader.len() as u64);
                        self.writer_then(reader, spectrum_count)?;
                    } else {
                        let reader = MzMLReaderType::open_path(self.input_file.clone())?;
                        let spectrum_count = Some(reader.len() as u64);
                        self.writer_then(reader, spectrum_count)?;
                    }
                }
                #[cfg(feature = "mzmlb")]
                MassSpectrometryFormat::MzMLb => {
                    let reader = MzMLbReaderType::open_path(self.input_file.clone())?;
                    let spectrum_count = Some(reader.len() as u64);
                    self.writer_then(reader, spectrum_count)?;
                }
                _ => {
                    return Err(MZDeisotoperError::FormatUnknownOrNotSupportedError(
                        self.input_file.clone(),
                        ms_format,
                    ))
                }
            }
        }
        Ok(())
    }

    fn writer_then<
        R: RandomAccessSpectrumIterator<CPeak, DPeak, SpectrumType>
            + MSDataFileMetadata
            + Send
            + 'static,
    >(
        &self,
        reader: R,
        spectrum_count: Option<u64>,
    ) -> Result<(), MZDeisotoperError> {
        if self.output_file == PathBuf::from("-") {
            let outfile = io::stdout();
            let mut writer = MzMLWriterType::<_, CPeak, DPeak>::new(outfile);
            writer.copy_metadata_from(&reader);
            if let Some(spectrum_count) = spectrum_count {
                writer.set_spectrum_count(spectrum_count);
            }
            self.run_workflow(reader, writer)?;
        } else {
            let (ms_format, compressed) = infer_from_path(&self.output_file);
            match ms_format {
                MassSpectrometryFormat::MGF => {
                    let handle = io::BufWriter::new(fs::File::create(self.output_file.clone())?);
                    if compressed {
                        let encoder = GzEncoder::new(handle, Compression::best());
                        let writer = MGFWriterType::new(encoder);
                        self.run_workflow(reader, writer)?
                    } else {
                        let writer = MGFWriterType::new(handle);
                        self.run_workflow(reader, writer)?
                    }
                }
                MassSpectrometryFormat::MzML => {
                    let handle = io::BufWriter::new(fs::File::create(self.output_file.clone())?);
                    if compressed {
                        let encoder = GzEncoder::new(handle, Compression::best());
                        let mut writer = MzMLWriterType::new(encoder);
                        writer.copy_metadata_from(&reader);
                        if let Some(spectrum_count) = spectrum_count {
                            writer.set_spectrum_count(spectrum_count);
                        }
                        self.run_workflow(reader, writer)?;
                    } else {
                        let mut writer = MzMLWriterType::new(handle);
                        writer.copy_metadata_from(&reader);
                        if let Some(spectrum_count) = spectrum_count {
                            writer.set_spectrum_count(spectrum_count);
                        }
                        self.run_workflow(reader, writer)?;
                    }
                }
                #[cfg(feature = "mzmlb")]
                MassSpectrometryFormat::MzMLb => {
                    let mut builder = MzMLbWriterBuilder::new(self.output_file.clone());
                    if let Ok(value) = env::var("MZDEIOSTOPE_BLOSC_ZSTD") {
                        log::warn!("Non-standard Blosc compression was requested via MZDEIOSTOPE_BLOSC_ZSTD env-var");
                        MzMLbReaderType::<CPeak, DPeak>::set_blosc_nthreads(4);
                        builder = builder.with_blosc_zstd_compression(value.parse().unwrap());
                    } else {
                        builder = builder.with_zlib_compression(9);
                    }
                    let mut writer = builder.create()?;
                    // let mut writer = MzMLbWriterType::new(&self.output_file)?;
                    writer.copy_metadata_from(&reader);
                    if let Some(spectrum_count) = spectrum_count {
                        writer.set_spectrum_count(spectrum_count);
                    }
                    self.run_workflow(reader, writer)?;
                }
                _ => {
                    return Err(MZDeisotoperError::OutputFormatUnknownOrNotSupportedError(
                        self.output_file.to_string_lossy().to_string(),
                        ms_format,
                    ))
                }
            }
        }
        Ok(())
    }

    fn run_workflow<
        R: RandomAccessSpectrumIterator<CPeak, DPeak, SpectrumType>
            + MSDataFileMetadata
            + Send
            + 'static,
        W: ScanWriter<'static, CPeak, DPeak> + Send + 'static,
    >(
        &self,
        reader: R,
        writer: W,
    ) -> io::Result<()> {
        let buffer_size = 2000;
        let (send_solved, recv_solved) = sync_channel(buffer_size);
        let (send_collated, recv_collated) = sync_channel(buffer_size);

        let mut ms1_args = make_default_ms1_deconvolution_params();
        let mut signal_params = make_default_signal_processing_params();
        let mut msn_args = make_default_msn_deconvolution_params();

        ms1_args.fit_filter.threshold = self.ms1_score_threshold;
        ms1_args.isotopic_model = self.ms1_isotopic_model.into();
        ms1_args.charge_range = self.charge_range.into();
        ms1_args.max_missed_peaks = self.ms1_missed_peaks;

        msn_args.fit_filter.threshold = self.msn_score_threshold;
        msn_args.isotopic_model = self.msn_isotopic_model.into();
        msn_args.charge_range = self.charge_range.into();
        msn_args.max_missed_peaks = self.msn_missed_peaks;

        signal_params.ms1_averaging = self.ms1_averaging_range as usize;
        signal_params.ms1_denoising = self.ms1_denoising;

        let rt_range = self.time_range;
        let precursor_processing = self.precursor_processing;

        let start = Instant::now();
        let read_task = thread::spawn(move || {
            prepare_procesing(
                reader,
                ms1_args,
                msn_args,
                signal_params,
                send_solved,
                rt_range,
                Some(precursor_processing),
            )
        });

        let collate_task =
            thread::spawn(move || collate_results_spectra(recv_solved, send_collated));

        let write_task = thread::spawn(move || write_output_spectra(writer, recv_collated));

        match read_task.join() {
            Ok(o) => {
                let prog = o?;
                log::info!("MS1 Spectra: {}", prog.ms1_spectra);
                log::info!("MSn Spectra: {}", prog.msn_spectra);
                log::info!(
                    "Precursors Defaulted: {} | Mismatched Charge State: {}",
                    prog.precursors_defaulted,
                    prog.precursor_charge_state_mismatch
                );
                log::info!("MS1 Peaks: {}", prog.ms1_peaks);
                log::info!("MSn Peaks: {}", prog.msn_peaks);
            }
            Err(e) => {
                log::warn!("Failed to join reader task: {e:?}");
            }
        }
        let read_done = Instant::now();
        let processing_elapsed = read_done - start;

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

        let done = Instant::now();
        let elapsed = done - start;
        if (elapsed.as_secs_f64() - processing_elapsed.as_secs_f64()) > 2.0 {
            log::info!("Total Elapsed Time: {:0.3?}", elapsed);
        }
        Ok(())
    }
}

fn main() -> Result<(), MZDeisotoperError> {
    pretty_env_logger::init_timed();

    let args = MZDeiosotoperArgs::parse();
    args.main()?;
    Ok(())
}
