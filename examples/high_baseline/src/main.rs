use mzdata::{
    io::{mzml::MzMLWriterType, MZReaderType},
    params::ControlledVocabulary,
    prelude::*,
    spectrum::SignalContinuity
};
use mzpeaks::CentroidPeak;
use std::{io, time::Instant};

use mzdeisotope::{
    deconvolute_peaks,
    isotopic_model::{IsotopicModels, IsotopicPatternParams, PROTON},
    scorer::{MaximizingFitFilter, PenalizedMSDeconvScorer},
};

use tracing::info;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use clap::Parser;

#[derive(Parser)]
struct App {
    #[arg()]
    path: String,

    #[arg(short, long, default_value_t = 0)]
    index: usize,

    #[arg(short, long, default_value_t = 3.0)]
    background_reduction: f32,
}

fn prepare_writer(
    reader: &MZReaderType<std::fs::File, CentroidPeak, mzdeisotope::DeconvolvedSolutionPeak>,
) -> MzMLWriterType<io::StdoutLock<'_>, CentroidPeak, mzdeisotope::DeconvolvedSolutionPeak> {
    let mut writer = MzMLWriterType::new(io::stdout().lock());
    writer.copy_metadata_from(reader);

    let source_file = writer
        .file_description_mut()
        .source_files
        .get_mut(0)
        .unwrap();
    let ids_to_remove: Vec<_> = source_file
        .iter_params()
        .enumerate()
        .filter(|(_, p)| p.name().contains("nativeID"))
        .map(|(i, _)| i)
        .collect();
    for (i, j) in ids_to_remove.into_iter().enumerate() {
        source_file.remove_param(j - i);
    }
    source_file.id_format =
        Some(ControlledVocabulary::MS.param(1000774, "multiple peak list nativeID format"));
    writer.set_spectrum_count(3);
    writer
}

fn configure_log() {
    let subscriber = tracing_subscriber::registry()
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::TRACE.into()))
        .with(
            fmt::layer()
                .compact()
                .with_timer(fmt::time::ChronoLocal::rfc_3339())
                .with_writer(io::stderr)
                .with_filter(
                    EnvFilter::builder()
                        .with_default_directive(tracing::Level::INFO.into())
                        .from_env_lossy(),
                ),
        );

    subscriber.init();
}

fn main() -> io::Result<()> {
    configure_log();

    let args = App::parse();

    let path = args.path;
    let scan_index: usize = args.index;

    info!("Opening {path} and processing index {scan_index}");

    let mut reader: MZReaderType<std::fs::File, _, mzdeisotope::DeconvolvedSolutionPeak> =
        MZReaderType::open_path(path)?;

    let mut scan = reader.get_spectrum_by_index(scan_index).unwrap();

    let mut writer = prepare_writer(&reader);

    scan.description_mut().id = "index=1".to_string();
    writer.write(&scan)?;

    let tic_before = scan.peaks().tic();
    scan.denoise(args.background_reduction).unwrap();
    let tic_after = scan.peaks().tic();

    info!(
        "TIC before: {tic_before}, TIC after: {tic_after}. Ratio = {}",
        tic_after / tic_before
    );

    scan.description_mut().id = "index=2".to_string();
    writer.write(&scan)?;

    scan.pick_peaks(1.5).unwrap();

    let peaks = scan.peaks.clone().unwrap();
    info!("Picked {} peaks", peaks.len());

    let start_time = Instant::now();

    let deconv_peaks = deconvolute_peaks(
        peaks,
        IsotopicModels::Peptide,
        Tolerance::PPM(20.0),
        (2, 20),
        PenalizedMSDeconvScorer::new(0.02, 2.0),
        MaximizingFitFilter::new(7.0),
        1,
        IsotopicPatternParams::new(0.999, 0.001, Some(0.85), PROTON),
        true,
    )
    .unwrap();
    let elapsed = Instant::now() - start_time;
    info!(
        "Deconvolved {} peaks in {:0.3} seconds",
        deconv_peaks.len(),
        elapsed.as_secs_f64()
    );

    scan.description_mut().id = "index=3".to_string();
    scan.description_mut().signal_continuity = SignalContinuity::Centroid;
    scan.deconvoluted_peaks = Some(deconv_peaks);
    writer.write(&scan)?;

    Ok(())
}
