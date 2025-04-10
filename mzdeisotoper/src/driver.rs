use std::fs;
use std::io;
use std::path::PathBuf;
use std::str::FromStr;
use std::thread;
use std::time::Instant;

use crossbeam_channel::bounded;

use clap::Parser;
use serde::{Deserialize, Serialize};

use flate2::write::GzEncoder;
use flate2::Compression;

use thiserror::Error;

use mzdata::meta::{
    custom_software_name, DataProcessing, DataProcessingAction, ProcessingMethod, Software,
};
use mzdata::params::Param;

use tracing::{debug, info, warn};

#[cfg(feature = "mzmlb")]
use mzdata::io::mzmlb::{MzMLbReaderType, MzMLbWriterBuilder};
#[cfg(feature = "bruker_tdf")]
use mzdata::io::tdf::TDFSpectrumReaderType;
#[cfg(feature = "thermo")]
use mzdata::io::thermo::ThermoRawReaderType;
use mzdata::io::{
    infer_format, infer_from_path, infer_from_stream,
    mgf::{MGFReaderType, MGFWriterType},
    mzml::{MzMLReaderType, MzMLWriterType},
    MassSpectrometryFormat, PreBufferedStream, RestartableGzDecoder, StreamingSpectrumIterator,
};
use mzdata::prelude::*;
#[cfg(feature = "mzmlb")]
use std::env;

use mzdeisotope::scorer::ScoreType;

use crate::args::{
    make_default_ms1_deconvolution_params, make_default_msn_deconvolution_params, ArgChargeRange,
    ArgIsotopicModels, PrecursorProcessing, SignalParams,
};
use crate::{
    make_default_ms1_feature_extraction_params, make_default_msn_feature_extraction_params,
    proc::{prepare_procesing, prepare_procesing_im},
    time_range::TimeRange,
    types::{CFeature, CPeak, DFeature, DPeak, FrameType, SpectrumType, BUFFER_SIZE},
    write::{collate_results_spectra, write_output_frames, write_output_spectra},
};

fn non_negative_float_f32(s: &str) -> Result<f32, String> {
    let value = s.parse::<f32>().map_err(|e| e.to_string())?;
    if value < 0.0 {
        Err(format!("`{s}` is less than zero"))
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

    #[error("The input file format does not support ion mobility when it was requested")]
    FormatDoesNotSupportIonMobility(MassSpectrometryFormat),
    #[error(
        "The input file {0} in {1} format did not contain ion mobility data when it was requested"
    )]
    FileDoesNotContainIonMobility(String, MassSpectrometryFormat),
}

/// Deisotoping and charge state deconvolution of mass spectrometry files.
///
/// Read a file or stream, transform the spectra, and write out a processed mzML or MGF
/// file, or an mzML stream.
#[derive(Parser, Debug, Deserialize, Serialize)]
#[command(author, version)]
pub struct MZDeiosotoper {
    /// The path to read the input spectra from, or if '-' is passed, read from STDIN
    #[arg()]
    pub input_file: String,

    /// The path to write the output file to, or if '-' is passed, write to STDOUT.
    ///
    /// If a path is specified, the output format is inferred, otherwise mzML is assumed.
    #[arg(short = 'o', long = "output-file", default_value = "-")]
    pub output_file: PathBuf,

    /// The path to write a log file to, in addition to STDERR
    #[arg(short = 'l', long = "log-file", help_heading = "Config")]
    pub log_file: Option<PathBuf>,

    /// A TOML configuration file to read additional parameters from.
    ///
    /// Configurations are also read from `mzdeisotoper.toml` in the working directory.
    /// Environment variables prefixed with `MZDEISOTOPER_` will be read too.
    #[arg(long = "config-file", help_heading = "Config")]
    pub config_file: Option<PathBuf>,

    /// Write TOML configuration to this file
    #[arg(long, help_heading = "Config")]
    pub write_config_file: Option<PathBuf>,

    /// The size of the buffer for queueing writing of results to the output stream.
    ///
    /// Making this larger consumes more memory but reduces the odds of bottlenecks
    /// forming due to a spike in file system delays.
    #[arg(short = 'w', long="write-buffer-size", default_value_t=BUFFER_SIZE)]
    pub write_buffer_size: usize,

    /// The number of threads to use, passing a value < 1 to use all available threads
    #[arg(
        short='t',
        long="threads",
        default_value_t=-1,
    )]
    pub threads: i32,

    /// The time range to process, denoted (start?)-(stop?)
    #[arg(
        short='r',
        long="time-range",
        value_parser=TimeRange::from_str,
        value_name="BEGIN-END",
        long_help=r#"The time range to process, denoted (start?)-(stop?)

If a start is not specified, processing begins from the start of the run.
If a stop is not specified, processing stops at the end of the run.
"#,
        help_heading = "Processing"
    )]
    pub time_range: Option<TimeRange>,

    /// The number of MS1 spectra before and after to average with prior to peak picking
    #[arg(
        short = 'g',
        long = "ms1-averaging-range",
        default_value_t = 0,
        value_parser = clap::value_parser!(u32).range(0..),
        help_heading = "Processing"
    )]
    pub ms1_averaging_range: u32,

    /// The magnitude of background noise reduction to use on MS1 spectra prior to peak picking
    #[arg(
        short = 'b',
        long = "ms1-background-reduction",
        default_value_t = 0.0,
        value_parser = non_negative_float_f32,
        help_heading = "Processing"
    )]
    pub ms1_denoising: f32,

    /// The isotopic model to use for MS1 spectra
    #[arg(short = 'a', long = "ms1-isotopic-model", default_value = "peptide", help_heading = "Processing")]
    pub ms1_isotopic_model: Vec<ArgIsotopicModels>,

    /// The minimum isotopic pattern fit score for MS1 spectra
    #[arg(short = 's', long = "ms1-score-threshold", default_value_t = 20.0, help_heading = "Processing")]
    pub ms1_score_threshold: ScoreType,

    /// The isotopic model to use for MSn spectra
    #[arg(short = 'A', long = "msn-isotopic-model", default_value = "peptide", help_heading = "Processing")]
    pub msn_isotopic_model: Vec<ArgIsotopicModels>,

    /// The minimum isotopic pattern fit score for MSn spectra
    #[arg(short = 'S', long = "msn-score-threshold", default_value_t = 10.0, help_heading = "Processing")]
    pub msn_score_threshold: ScoreType,

    #[arg(
        short = 'v',
        long = "precursor-processing",
        default_value = "selected-precursors",
        help = "How to treat precursor ranges",
        help_heading = "Processing"
    )]
    pub precursor_processing: PrecursorProcessing,

    /// The range of charge states to consider for each peak denoted (low)-(high) or (high)
    #[arg(
        short = 'z',
        long = "charge-range",
        default_value_t=ArgChargeRange(1, 8),
        help_heading = "Processing"
    )]
    pub charge_range: ArgChargeRange,

    /// The maximum number of missed peaks for MS1 spectra
    #[arg(short = 'm', long = "max-missed-peaks", default_value_t = 1, help_heading = "Processing")]
    pub ms1_missed_peaks: u16,

    /// The maximum number of missed peaks for MSn spectra
    #[arg(short = 'M', long = "msn-max-missed-peaks", default_value_t = 1, help_heading = "Processing")]
    pub msn_missed_peaks: u16,

    /// The mass error to tolerate for MS1 spectra
    #[arg(short = 'e', long = "ms1-mass-error-tolerance", help_heading = "Processing")]
    pub ms1_error_tolerance: Option<Tolerance>,

    /// The mass error to tolerate for MSn spectra
    #[arg(short = 'E', long = "msn-mass-error-tolerance", help_heading = "Processing")]
    pub msn_error_tolerance: Option<Tolerance>,

    /// Use incremental truncation of isotopic patterns instead of a single width
    #[arg(short = 'i', long = "incremental-truncation", help_heading = "Processing")]
    pub isotopic_incremental_truncation: bool,

    #[arg(
        skip,
        // help = "Specifies additional granular information about low-level signal processing operations"
    )]
    pub signal_params: SignalParams,

    /// Whether to use the ion mobility framing algorithm
    #[arg(short = 'q', long, default_value_t = false, help_heading = "Ion Mobility")]
    pub ion_mobility_deconvolution: bool,

    /// The maximum gap size in ion mobility units to tolerate in MS1 IM-MS features
    #[arg(long, default_value_t = 0.025, help_heading = "Ion Mobility")]
    pub ms1_ion_mobility_gap_size: f64,

    /// The maximum gap size in ion mobility units to tolerate in MSn IM-MS features
    #[arg(long, default_value_t = 0.025, help_heading = "Ion Mobility")]
    pub msn_ion_mobility_gap_size: f64,

    /// The minimum feature size for MS1 IM-MS features
    #[arg(short='f', long, default_value_t = 3, help_heading = "Ion Mobility")]
    pub ms1_ion_mobility_feature_min_length: usize,

    /// The minimum feature size for MSn IM-MS features
    #[arg(short='F', long, default_value_t = 2, help_heading = "Ion Mobility")]
    pub msn_ion_mobility_feature_min_length: usize,
}

impl MZDeiosotoper {
    fn create_threadpool(&self) -> rayon::ThreadPool {
        let num_threads = if self.threads > 0 {
            self.threads as usize
        } else {
            thread::available_parallelism().unwrap().into()
        };
        debug!("Using {} cores", num_threads);
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| {
                format!("worker-{i}")
            })
            .build()
            .unwrap()
    }

    fn make_software(&self) -> Software {
        let mut sw = Software::default();
        let name = custom_software_name("mzdeisotoper");
        sw.add_param(name);
        sw.id = "mzdeisotoper".to_string();
        sw.version = option_env!("CARGO_PKG_VERSION")
            .unwrap_or("unknown")
            .to_string();
        sw
    }

    fn make_processing_method(&self) -> ProcessingMethod {
        let mut processing = ProcessingMethod {
            software_reference: "mzdeisotoper".into(),
            ..Default::default()
        };
        processing.add_param(DataProcessingAction::Deisotoping.as_param_const().into());
        processing.add_param(
            DataProcessingAction::ChargeDeconvolution
                .as_param_const()
                .into(),
        );
        processing.add_param(DataProcessingAction::PeakPicking.as_param_const().into());
        processing.add_param(
            DataProcessingAction::ChargeStateCalculation
                .as_param_const()
                .into(),
        );
        processing.add_param(Param::new_key_value(
            "ms1_averaging_range",
            self.ms1_averaging_range,
        ));
        processing.add_param(Param::new_key_value("ms1_denoising", self.ms1_denoising));
        for m in self.ms1_isotopic_model.iter() {
            processing.add_param(Param::new_key_value("ms1_isotopic_model", m.to_string()));
        }
        for m in self.msn_isotopic_model.iter() {
            processing.add_param(Param::new_key_value("msn_isotopic_model", m.to_string()));
        }
        processing.add_param(Param::new_key_value(
            "ms1_score_threshold",
            self.ms1_score_threshold,
        ));
        processing.add_param(Param::new_key_value(
            "msn_score_threshold",
            self.msn_score_threshold,
        ));
        processing.add_param(Param::new_key_value(
            "precursor_processing",
            self.precursor_processing.to_string(),
        ));
        processing.add_param(Param::new_key_value(
            "charge_range",
            self.charge_range.to_string(),
        ));
        processing.add_param(Param::new_key_value(
            "ms1_missed_peaks",
            self.ms1_missed_peaks,
        ));
        processing.add_param(Param::new_key_value(
            "msn_missed_peaks",
            self.msn_missed_peaks,
        ));
        if self.isotopic_incremental_truncation {
            processing.add_param(Param::new_key_value(
                "isotopic_incremental_truncation",
                true,
            ))
        }
        processing.order = i8::MAX;
        processing
    }

    fn update_data_processing<T: MSDataFileMetadata>(&self, source: &mut T) {
        let sw_id = {
            let mut sw = self.make_software();
            let stem = sw.id.clone();
            let mut i = 0;
            let mut query = stem.clone();
            while source.softwares().iter().any(|s| s.id == query) {
                i += 1;
                query = format!("{stem}_{i}");
            }
            sw.id = query.clone();
            source.softwares_mut().push(sw);
            query
        };
        if source.data_processings().is_empty() {
            let mut method = self.make_processing_method();
            method.order = 0;
            method.software_reference = sw_id.clone();
            let mut dp = DataProcessing::default();
            let dp_id = "DP1_mzdeisotoper".to_string();
            dp.id = dp_id.clone();
            dp.push(method);
            source.data_processings_mut().push(dp);
            if let Some(descr) = source.run_description_mut() {
                descr.default_data_processing_id = Some(dp_id.clone());
            }
        } else {
            for dp in source.data_processings_mut().iter_mut() {
                let last_step = dp
                    .iter()
                    .max_by(|a, b| a.order.cmp(&b.order))
                    .map(|m| m.order)
                    .unwrap_or(-1);
                let mut method = self.make_processing_method();
                method.order = last_step + 1;
                method.software_reference = sw_id.clone();
                dp.push(method)
            }
        }
    }

    pub fn main(&self) -> Result<(), MZDeisotoperError> {
        info!(
            "mzdeisotoper v{}",
            option_env!("CARGO_PKG_VERSION").unwrap_or("unknown")
        );
        info!("Input: {}", self.input_file);
        info!("Output: {}", self.output_file.display());

        if let Some(path) = self.write_config_file.as_ref() {
            debug!("Writing config file to {}", path.display());
            let config_text = toml::to_string_pretty(&self).unwrap();
            fs::write(path, config_text)?;
        }

        // self.create_threadpool().install(|| self.reader_then())
        self.reader_then()
    }

    fn reader_then(&self) -> Result<(), MZDeisotoperError> {
        if self.input_file == "-" {
            let mut buffered =
                PreBufferedStream::new_with_buffer_size(io::stdin(), 2usize.pow(20))?;
            let (ms_format, compressed) = infer_from_stream(&mut buffered)?;
            debug!("Detected {ms_format:?} from STDIN (compressed? {compressed})");
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
            debug!("Inferring format from {}", self.input_file);
            let (ms_format, compressed) = infer_format(&self.input_file)?;
            debug!("Detected {ms_format:?} from path (compressed? {compressed})");
            match ms_format {
                MassSpectrometryFormat::MGF => {
                    if self.ion_mobility_deconvolution {
                        return Err(MZDeisotoperError::FormatDoesNotSupportIonMobility(
                            ms_format,
                        ));
                    }
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
                        if self.ion_mobility_deconvolution {
                            let reader: MzMLReaderType<
                                RestartableGzDecoder<io::BufReader<fs::File>>,
                                CPeak,
                                DPeak,
                            > = MzMLReaderType::new_indexed(fh);
                            let reader = reader.into_frame_source();
                            let spectrum_count = reader.spectrum_count_hint();
                            self.writer_ion_mobility(reader, spectrum_count)?;
                        } else {
                            let reader = StreamingSpectrumIterator::new(MzMLReaderType::new(fh));
                            let spectrum_count = Some(reader.len() as u64);
                            self.writer_then(reader, spectrum_count)?;
                        }
                    } else {
                        let reader = MzMLReaderType::open_path(self.input_file.clone())?;
                        if self.ion_mobility_deconvolution {
                            match reader.try_into_frame_source::<CFeature, DFeature>() {
                                Ok(reader) => {
                                    let spectrum_count = reader.spectrum_count_hint();
                                    self.writer_ion_mobility(reader, spectrum_count)?;
                                }
                                Err(_) => {
                                    return Err(MZDeisotoperError::FileDoesNotContainIonMobility(
                                        self.input_file.clone(),
                                        ms_format,
                                    ))
                                }
                            }
                        } else {
                            let spectrum_count = Some(reader.len() as u64);
                            self.writer_then(reader, spectrum_count)?;
                        }
                    }
                }
                #[cfg(feature = "mzmlb")]
                MassSpectrometryFormat::MzMLb => {
                    let reader = MzMLbReaderType::open_path(self.input_file.clone())?;
                    let spectrum_count = Some(reader.len() as u64);
                    self.writer_then(reader, spectrum_count)?;
                }
                #[cfg(feature = "thermo")]
                MassSpectrometryFormat::ThermoRaw => {
                    let reader = ThermoRawReaderType::open_path(self.input_file.clone())?;
                    let spectrum_count = Some(reader.len() as u64);
                    self.writer_then(reader, spectrum_count)?;
                }
                #[cfg(feature = "bruker_tdf")]
                MassSpectrometryFormat::BrukerTDF => {
                    let reader: TDFSpectrumReaderType<CFeature, DFeature, CPeak, DPeak> =
                        TDFSpectrumReaderType::open_path(self.input_file.clone())?;
                    if self.ion_mobility_deconvolution {
                        match reader.try_into_frame_source::<CFeature, DFeature>() {
                            Ok(reader) => {
                                let spectrum_count = reader.spectrum_count_hint();
                                self.writer_ion_mobility(reader, spectrum_count)?;
                            }
                            Err(_) => {
                                return Err(MZDeisotoperError::FileDoesNotContainIonMobility(
                                    self.input_file.clone(),
                                    ms_format,
                                ))
                            }
                        }
                    } else {
                        todo!("Handling of TDF data without ion mobility not yet implemented")
                    }
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
            self.update_data_processing(&mut writer);
            if let Some(spectrum_count) = spectrum_count {
                writer.set_spectrum_count(spectrum_count);
            }
            self.run_workflow(reader, writer, MassSpectrometryFormat::MzML)?;
        } else {
            let (ms_format, compressed) = infer_from_path(&self.output_file);
            match ms_format {
                MassSpectrometryFormat::MGF => {
                    let handle = io::BufWriter::new(fs::File::create(self.output_file.clone())?);
                    if compressed {
                        let encoder = GzEncoder::new(handle, Compression::best());
                        let writer = MGFWriterType::new(encoder);
                        self.run_workflow(reader, writer, MassSpectrometryFormat::MGF)?
                    } else {
                        let writer = MGFWriterType::new(handle);
                        self.run_workflow(reader, writer, MassSpectrometryFormat::MGF)?
                    }
                }
                MassSpectrometryFormat::MzML => {
                    let handle = io::BufWriter::new(fs::File::create(self.output_file.clone())?);
                    if compressed {
                        let encoder = GzEncoder::new(handle, Compression::best());
                        let mut writer = MzMLWriterType::new(encoder);
                        writer.copy_metadata_from(&reader);
                        self.update_data_processing(&mut writer);
                        if let Some(spectrum_count) = spectrum_count {
                            writer.set_spectrum_count(spectrum_count);
                        }
                        self.run_workflow(reader, writer, MassSpectrometryFormat::MzML)?;
                    } else {
                        let mut writer = MzMLWriterType::new(handle);
                        writer.copy_metadata_from(&reader);
                        self.update_data_processing(&mut writer);
                        if let Some(spectrum_count) = spectrum_count {
                            writer.set_spectrum_count(spectrum_count);
                        }
                        self.run_workflow(reader, writer, MassSpectrometryFormat::MzML)?;
                    }
                }
                #[cfg(feature = "mzmlb")]
                MassSpectrometryFormat::MzMLb => {
                    let mut builder = MzMLbWriterBuilder::new(self.output_file.clone());
                    if let Ok(value) = env::var("MZDEIOSTOPE_BLOSC_ZSTD") {
                        warn!("Non-standard Blosc compression was requested via MZDEIOSTOPE_BLOSC_ZSTD env-var");
                        MzMLbReaderType::<CPeak, DPeak>::set_blosc_nthreads(4);
                        builder = builder.with_blosc_zstd_compression(value.parse().unwrap());
                    } else {
                        builder = builder.with_zlib_compression(9);
                    }
                    let mut writer = builder.create()?;
                    // let mut writer = MzMLbWriterType::new(&self.output_file)?;
                    writer.copy_metadata_from(&reader);
                    self.update_data_processing(&mut writer);
                    if let Some(spectrum_count) = spectrum_count {
                        writer.set_spectrum_count(spectrum_count);
                    }
                    self.run_workflow(reader, writer, MassSpectrometryFormat::MzMLb)?;
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

    fn writer_ion_mobility<
        R: RandomAccessIonMobilityFrameIterator<CFeature, DFeature, FrameType>
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
            self.update_data_processing(&mut writer);
            if let Some(spectrum_count) = spectrum_count {
                writer.set_spectrum_count(spectrum_count);
            }
            self.run_workflow_ion_mobility(reader, writer, MassSpectrometryFormat::MzML)?;
        } else {
            let (ms_format, compressed) = infer_from_path(&self.output_file);
            match ms_format {
                MassSpectrometryFormat::MGF => {
                    todo!("Have not implemented ion mobility writing for MGF");
                    // let handle = io::BufWriter::new(fs::File::create(self.output_file.clone())?);
                    // if compressed {
                    //     let encoder = GzEncoder::new(handle, Compression::best());
                    //     let writer = MGFWriterType::new(encoder);
                    //     self.run_workflow_ion_mobility(reader, writer, MassSpectrometryFormat::MGF)?
                    // } else {
                    //     let writer = MGFWriterType::new(handle);
                    //     self.run_workflow_ion_mobility(reader, writer, MassSpectrometryFormat::MGF)?
                    // }
                }
                MassSpectrometryFormat::MzML => {
                    let handle = io::BufWriter::new(fs::File::create(self.output_file.clone())?);
                    if compressed {
                        let encoder = GzEncoder::new(handle, Compression::best());
                        let mut writer = MzMLWriterType::new(encoder);
                        writer.copy_metadata_from(&reader);
                        self.update_data_processing(&mut writer);
                        if let Some(spectrum_count) = spectrum_count {
                            writer.set_spectrum_count(spectrum_count);
                        }
                        self.run_workflow_ion_mobility(
                            reader,
                            writer,
                            MassSpectrometryFormat::MzML,
                        )?;
                    } else {
                        let mut writer = MzMLWriterType::new(handle);
                        writer.copy_metadata_from(&reader);
                        self.update_data_processing(&mut writer);
                        if let Some(spectrum_count) = spectrum_count {
                            writer.set_spectrum_count(spectrum_count);
                        }
                        self.run_workflow_ion_mobility(
                            reader,
                            writer,
                            MassSpectrometryFormat::MzML,
                        )?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn run_workflow<
        R: RandomAccessSpectrumIterator<CPeak, DPeak, SpectrumType>
            + MSDataFileMetadata
            + Send
            + 'static,
        W: SpectrumWriter<CPeak, DPeak> + Send + 'static,
    >(
        &self,
        reader: R,
        writer: W,
        writer_format: MassSpectrometryFormat,
    ) -> io::Result<()> {
        let buffer_size = self.write_buffer_size;

        let mut signal_params = self.signal_params;
        let mut ms1_args = make_default_ms1_deconvolution_params();
        let mut msn_args = make_default_msn_deconvolution_params();
        ms1_args.mz_range = signal_params.mz_range;
        msn_args.mz_range = signal_params.mz_range;

        ms1_args.fit_filter.threshold = self.ms1_score_threshold;
        ms1_args.isotopic_model = self
            .ms1_isotopic_model
            .iter()
            .map(|it| (*it).into())
            .collect();
        ms1_args.charge_range = self.charge_range.into();
        ms1_args.max_missed_peaks = self.ms1_missed_peaks;

        if self.isotopic_incremental_truncation {
            ms1_args.isotopic_params.incremental_truncation = Some(0.95);
            ms1_args.isotopic_params.truncate_after = 0.9999;
            msn_args.isotopic_params.incremental_truncation = Some(0.8);
            msn_args.isotopic_params.truncate_after = 0.9999;
        }

        msn_args.fit_filter.threshold = self.msn_score_threshold;
        msn_args.isotopic_model = self
            .msn_isotopic_model
            .iter()
            .map(|it| (*it).into())
            .collect();
        msn_args.charge_range = self.charge_range.into();
        msn_args.max_missed_peaks = self.msn_missed_peaks;

        signal_params.ms1_averaging = self.ms1_averaging_range as usize;
        signal_params.ms1_denoising = self.ms1_denoising;

        let rt_range = self.time_range;
        let precursor_processing = self.precursor_processing;

        let start = Instant::now();

        let (send_solved, recv_solved) = bounded(buffer_size);
        let (send_collated, recv_collated) = bounded(buffer_size);

        let worker_pool = self.create_threadpool();

        let read_task = thread::Builder::new().name("reader".into()).spawn(move || {
            worker_pool.install(|| {
                prepare_procesing(
                    reader,
                    ms1_args,
                    msn_args,
                    signal_params,
                    send_solved,
                    rt_range,
                    Some(precursor_processing),
                    writer_format,
                )
            })
        })?;

        let collate_task =
            thread::Builder::new().name("collater".into()).spawn(move || collate_results_spectra(recv_solved, send_collated))?;

        let write_task = thread::Builder::new().name("writer".into()).spawn(move || write_output_spectra(writer, recv_collated))?;

        match read_task.join() {
            Ok(o) => {
                let prog = o?;
                info!("MS1 Spectra: {}", prog.ms1_spectra);
                info!("MSn Spectra: {}", prog.msn_spectra);
                info!(
                    "Precursors Defaulted: {} | Mismatched Charge State: {}",
                    prog.precursors_defaulted, prog.precursor_charge_state_mismatch
                );
                info!("MS1 Peaks: {}", prog.ms1_peaks);
                info!("MSn Peaks: {}", prog.msn_peaks);
            }
            Err(e) => {
                warn!("Failed to join reader task: {e:?}");
            }
        }
        let read_done = Instant::now();
        let processing_elapsed = read_done - start;

        match collate_task.join() {
            Ok(_) => {}
            Err(e) => {
                warn!("Failed to join collator task: {e:?}")
            }
        }

        match write_task.join() {
            Ok(o) => o?,
            Err(e) => {
                warn!("Failed to join writer task: {e:?}");
            }
        }

        let done = Instant::now();
        let elapsed = done - start;
        if (elapsed.as_secs_f64() - processing_elapsed.as_secs_f64()) > 2.0 {
            info!("Total Elapsed Time: {:0.3?}", elapsed);
        }
        Ok(())
    }

    fn run_workflow_ion_mobility<
        R: RandomAccessIonMobilityFrameIterator<CFeature, DFeature, FrameType>
            + MSDataFileMetadata
            + Send
            + 'static,
        W: IonMobilityFrameWriter<CFeature, DFeature> + SpectrumWriter<CPeak, DPeak> + Send + 'static,
    >(
        &self,
        reader: R,
        writer: W,
        writer_format: MassSpectrometryFormat,
    ) -> io::Result<()> {
        let buffer_size = self.write_buffer_size;
        info!("Running ion mobility frame workflow");

        let mut signal_params = self.signal_params;
        let mut ms1_args = make_default_ms1_deconvolution_params();
        let mut msn_args = make_default_msn_deconvolution_params();
        ms1_args.mz_range = signal_params.mz_range;
        msn_args.mz_range = signal_params.mz_range;

        ms1_args.fit_filter.threshold = self.ms1_score_threshold;
        ms1_args.isotopic_model = self
            .ms1_isotopic_model
            .iter()
            .map(|it| (*it).into())
            .collect();
        ms1_args.charge_range = self.charge_range.into();
        ms1_args.max_missed_peaks = self.ms1_missed_peaks;

        msn_args.fit_filter.threshold = self.msn_score_threshold;
        msn_args.isotopic_model = self
            .msn_isotopic_model
            .iter()
            .map(|it| (*it).into())
            .collect();
        msn_args.charge_range = self.charge_range.into();
        msn_args.max_missed_peaks = self.msn_missed_peaks;

        signal_params.ms1_averaging = self.ms1_averaging_range as usize;
        signal_params.ms1_denoising = self.ms1_denoising;

        let time_range = self.time_range;
        let precursor_processing = self.precursor_processing;

        let mut extraction_params = make_default_ms1_feature_extraction_params();
        extraction_params.smoothing = self.signal_params.ms1_averaging;
        extraction_params.maximum_time_gap = self.ms1_ion_mobility_gap_size;
        extraction_params.minimum_size = self.ms1_ion_mobility_feature_min_length;
        if let Some(tol) = self.ms1_error_tolerance {
            extraction_params.error_tolerance = tol;
        }

        let mut msn_extraction_params = make_default_msn_feature_extraction_params();
        msn_extraction_params.smoothing = self.signal_params.ms1_averaging;
        msn_extraction_params.maximum_time_gap = self.msn_ion_mobility_gap_size;
        msn_extraction_params.minimum_size = self.msn_ion_mobility_feature_min_length;
        if let Some(tol) = self.msn_error_tolerance {
            msn_extraction_params.error_tolerance = tol;
        }

        let start = Instant::now();

        let (send_solved, recv_solved) = bounded(buffer_size);
        let (send_collated, recv_collated) = bounded(buffer_size);

        let worker_pool = self.create_threadpool();

        let read_task = thread::Builder::new().name("reader-task".into()).spawn(move || {
            worker_pool.install(|| {
                prepare_procesing_im(
                    reader,
                    ms1_args,
                    msn_args,
                    extraction_params,
                    msn_extraction_params,
                    send_solved,
                    time_range,
                    Some(precursor_processing),
                    writer_format,
                )
            })
        })?;

        let collate_task =
            thread::Builder::new().name("collator-task".into()).spawn(move || collate_results_spectra(recv_solved, send_collated))?;

        let write_task = thread::Builder::new().name("writer-task".into()).spawn(move || write_output_frames(writer, recv_collated))?;

        match read_task.join() {
            Ok(o) => {
                let prog = o?;
                info!("MS1 Frames: {}", prog.ms1_spectra);
                info!("MSn Frames: {}", prog.msn_spectra);
                info!("MS1 Features: {}", prog.ms1_peaks);
                info!("MSn Features: {}", prog.msn_peaks);
            }
            Err(e) => {
                warn!("Failed to join reader task: {e:?}");
            }
        }
        let read_done = Instant::now();
        let processing_elapsed = read_done - start;

        match collate_task.join() {
            Ok(_) => {}
            Err(e) => {
                warn!("Failed to join collator task: {e:?}")
            }
        }

        match write_task.join() {
            Ok(o) => o?,
            Err(e) => {
                warn!("Failed to join writer task: {e:?}");
            }
        }

        let done = Instant::now();
        let elapsed = done - start;
        if (elapsed.as_secs_f64() - processing_elapsed.as_secs_f64()) > 2.0 {
            info!("Total Elapsed Time: {:0.3?}", elapsed);
        }

        Ok(())
    }
}
