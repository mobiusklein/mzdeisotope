use std::io;

use mzdata::prelude::*;
use mzpeaks::feature::Feature;
use mzpeaks::feature_map::FeatureMap;
use mzpeaks::{Mass, Time, Tolerance, MZ};
use mzsignal::feature_mapping::{FeatureExtracter, FeatureExtracterType};

use mzdeisotope::isotopic_model::{
    CachingIsotopicModel, IsotopicModels, TheoreticalIsotopicDistributionScalingMethod,
};
use mzdeisotope::scorer::{MaximizingFitFilter, PenalizedMSDeconvScorer};
use mzdeisotope_map::{DeconvolvedSolutionFeature, FeatureProcessor, FeatureSearchParams};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

fn prepare_feature_map() -> io::Result<FeatureMap<MZ, Time, Feature<MZ, Time>>> {
    let reader = mzdata::MZReader::open_path("../mzdeisotoper/tests/data/batching_test.mzML")?;

    let scans: Vec<_> = reader
        .into_iter()
        .filter(|s| s.ms_level() == 1 && s.start_time() < 122.0)
        .map(|mut s| {
            s.pick_peaks(1.0).unwrap();
            s
        })
        .collect();

    let mut extractor: FeatureExtracterType<_, _, _, Time> = FeatureExtracter::from_iter(
        scans
            .into_iter()
            .map(|s| (s.start_time(), s.peaks.unwrap())),
    );

    let features: FeatureMap<MZ, Time, Feature<MZ, Time>> =
        extractor.extract_features(Tolerance::PPM(10.0), 3, 0.25);
    Ok(features)
}

#[test]
fn test_map() -> io::Result<()> {
    let subscriber = tracing_subscriber::registry().with(
        fmt::layer().compact().with_writer(io::stderr).with_filter(
            EnvFilter::builder()
                .with_default_directive(tracing::Level::DEBUG.into())
                .from_env_lossy(),
        ),
    );
    // let log_file = fs::File::create("mzdeisotope-map.log")?;
    // let (log_file, _guard) = tracing_appender::non_blocking(log_file);
    // let subscriber = subscriber.with(
    //     fmt::layer()
    //         .compact()
    //         .with_ansi(false)
    //         .with_writer(log_file)
    //         .with_filter(
    //             EnvFilter::builder()
    //                 .with_default_directive(tracing::Level::DEBUG.into())
    //                 .from_env_lossy(),
    //         ),
    // );
    subscriber.init();

    let features = prepare_feature_map()?;
    tracing::debug!("{} raw features", features.len());
    tracing::debug!(
        "Max m/z = {}",
        features
            .iter()
            .map(|f| f.mz())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
    );

    let mut deconv = FeatureProcessor::new(
        features.clone(),
        CachingIsotopicModel::from(IsotopicModels::Glycopeptide),
        PenalizedMSDeconvScorer::new(0.02, 2.0),
        MaximizingFitFilter::new(10.0),
        TheoreticalIsotopicDistributionScalingMethod::Sum,
        3,
        0.25,
        true,
    );
    let params = FeatureSearchParams {
        truncate_after: 0.95,
        max_missed_peaks: 2,
        threshold_scale: 0.3,
        detection_threshold: 0.1,
    };
    let deconv_map = deconv
        .deconvolve(Tolerance::PPM(10.0), (1, 8), 1, 1, &params, 1e-3, 10)
        .unwrap();

    let features = deconv_map.all_features_for(4368.263, Tolerance::PPM(10.0));
    assert_eq!(features.len(), 2);

    fn query(mass: f64, feature_map: &FeatureMap<Mass, Time, DeconvolvedSolutionFeature<Time>>) {
        let mut hits: Vec<_> = feature_map
            .all_features_for(mass, Tolerance::PPM(20.0))
            .into_iter()
            .collect();
        hits.sort_by(|a, b| {
            a.charge()
                .cmp(&b.charge())
                .then_with(|| a.start_time().partial_cmp(&b.start_time()).unwrap())
        });
        tracing::debug!(
            "{} solved features, {} hits for query",
            feature_map.len(),
            hits.len()
        );
        for hit in hits {
            tracing::debug!(
                "{:0.2} {} ({:0.2}|{:0.2}) {}-{} {} points",
                hit.neutral_mass(),
                hit.charge(),
                hit.score,
                hit.intensity(),
                hit.start_time().unwrap(),
                hit.end_time().unwrap(),
                hit.len(),
            );

            for (i, (mass, time, inten)) in hit.iter().enumerate() {
                let envelope: Vec<_> = hit.envelope.iter().map(|e| e.at(i).unwrap()).collect();
                tracing::debug!("\t{i}\t{mass:0.3}@{time:0.2} => {inten:0.2} ({envelope:?})",);
            }
        }
    }

    query(4368.263, &deconv_map);
    Ok(())
}
