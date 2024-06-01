use std::io;

use mzdata::prelude::*;
use mzpeaks::{Time, Tolerance};
use mzsignal::feature_mapping::FeatureExtracter;

use mzdeisotope::isotopic_model::{
    CachingIsotopicModel, IsotopicModels, TheoreticalIsotopicDistributionScalingMethod,
};
use mzdeisotope::scorer::{MaximizingFitFilter, PenalizedMSDeconvScorer};
use mzdeisotope_map::{FeatureProcessor, FeatureSearchParams};
use tracing_subscriber::{fmt, EnvFilter, prelude::*};

#[test]
fn test_map() -> io::Result<()> {
    let subscriber = tracing_subscriber::registry()
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::TRACE.into()))
        .with(
            fmt::layer()
                .compact()
                .with_writer(io::stderr)
                .with_filter(
                    EnvFilter::builder()
                        .with_default_directive(tracing::Level::INFO.into())
                        .from_env_lossy(),
                ),
        );
    subscriber.init();
    let reader = mzdata::MZReader::open_path("../mzdeisotoper/tests/data/batching_test.mzML")?;

    let scans: Vec<_> = reader
        .into_iter()
        .filter(|s| s.ms_level() == 1)
        .map(|mut s| {
            s.pick_peaks(1.0).unwrap();
            s
        })
        .collect();

    let mut extractor: mzsignal::feature_mapping::FeatureExtracterType<_, _, _, Time> =
        FeatureExtracter::from_iter(
            scans
                .into_iter()
                .map(|s| (s.start_time(), s.peaks.unwrap())),
        );

    let features = extractor.extract_features(Tolerance::PPM(10.0), 3, 0.25);
    eprintln!("{} raw features", features.len());
    let mut deconv = FeatureProcessor::new(
        features,
        CachingIsotopicModel::from(IsotopicModels::Glycopeptide),
        PenalizedMSDeconvScorer::new(0.02, 1.0),
        MaximizingFitFilter::new(1.0),
        TheoreticalIsotopicDistributionScalingMethod::Sum,
        3,
        0.25,
        true,
    );
    let params = FeatureSearchParams {
        truncate_after: 0.95,
        max_missed_peaks: 1,
        threshold_scale: 0.3,
        detection_threshold: 0.1,
    };
    let deconv_map = deconv
        .deconvolve(Tolerance::PPM(10.0), (1, 8), 1, 1, &params, 1e-3, 10)
        .unwrap();
    eprintln!("{} solved features", deconv_map.len());
    Ok(())
}
