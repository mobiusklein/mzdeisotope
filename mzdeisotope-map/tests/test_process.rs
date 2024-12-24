use std::io;

use mzdata::prelude::*;
use mzdata::spectrum::bindata::BinaryArrayMap3D;

use mzpeaks::feature::Feature;
use mzpeaks::feature_map::FeatureMap;
use mzpeaks::{Mass, Time, Tolerance, MZ};
use mzsignal::feature_mapping::{FeatureExtracter, FeatureExtracterType};

use mzdeisotope::isotopic_model::{CachingIsotopicModel, IsotopicModels};
use mzdeisotope::scorer::{MaximizingFitFilter, PenalizedMSDeconvScorer};
use mzdeisotope_map::{
    solution::DeconvolvedSolutionFeature, FeatureProcessor, FeatureSearchParams,
};
use tracing::debug;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

fn prepare_feature_map() -> io::Result<FeatureMap<MZ, Time, Feature<MZ, Time>>> {
    let reader = mzdata::MZReader::open_path("../mzdeisotoper/tests/data/batching_test.mzML")?;

    let scans: Vec<_> = reader
        .into_iter()
        .filter(|s| s.ms_level() == 1 && s.start_time() < 125.0)
        .map(|mut s| {
            s.pick_peaks(1.0).unwrap();
            s
        })
        .collect();

    let n_peaks: usize = scans.iter().map(|s| s.peaks().len()).sum();
    debug!("Selected {} scans with {} peaks", scans.len(), n_peaks);

    let mut extractor: FeatureExtracterType<_, _, _, Time> = FeatureExtracter::from_iter(
        scans
            .into_iter()
            .map(|s| (s.start_time(), s.peaks.unwrap())),
    );

    let features: FeatureMap<MZ, Time, Feature<MZ, Time>> =
        extractor.extract_features(Tolerance::PPM(10.0), 3, 0.25);
    Ok(features)
}

fn init_logging() {
    let subscriber = tracing_subscriber::registry().with(
        fmt::layer().compact().with_writer(io::stderr).with_filter(
            EnvFilter::builder()
                .with_default_directive(tracing::Level::DEBUG.into())
                .from_env_lossy(),
        ),
    );
    // let log_file = std::fs::File::create("mzdeisotope-map.log")?;
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
}

#[allow(unused)]
fn write_feature_table<'a, Y, T: FeatureLike<MZ, Y> + 'a>(
    features: impl IntoIterator<Item = &'a T>,
    mut writer: impl io::Write,
) -> io::Result<()> {
    writeln!(writer, "feature_id,mz,time,intensity")?;
    for (i, feature) in features.into_iter().enumerate() {
        for (x, y, z) in feature.iter() {
            writeln!(writer, "{i},{x},{y},{z}")?
        }
    }
    Ok(())
}

#[allow(unused)]
fn write_3d_array(arrays: &BinaryArrayMap3D, mut writer: impl io::Write) -> io::Result<()> {
    writeln!(writer, "mz,time,intensity")?;
    for (time, arrays) in arrays.iter() {
        let mzs = arrays.mzs()?;
        let ints = arrays.intensities()?;
        for (mz, int) in mzs.into_iter().copied().zip(ints.into_iter().copied()) {
            writeln!(writer, "{mz},{time},{int}")?
        }
    }
    Ok(())
}



#[test]
fn test_map_im() -> io::Result<()> {
    init_logging();

    let sid = "merged=42926 frame=9728 scanStart=1 scanEnd=705";
    let mut frame = mzdata::mz_read!("../test/data/20200204_BU_8B8egg_1ug_uL_7charges_60_min_Slot2-11_1_244.mzML.gz".as_ref(), reader => {
        let mut reader = mzdata::io::Generic3DIonMobilityFrameSource::new(reader);
        let frame: mzdata::spectrum::MultiLayerIonMobilityFrame = reader.get_frame_by_id(sid).unwrap();
        frame
    })?;

    // write_3d_array(frame.arrays.as_ref().unwrap(), fs::File::create("./raw_arrays.csv")?)?;

    frame.extract_features_simple(Tolerance::PPM(15.0), 2, 0.1, None)?;
    frame.features.as_mut().unwrap().iter_mut().for_each(|f| {
        let sig = f.intensity_view();
        let mut smoothed_sig = sig.to_vec();
        mzsignal::smooth::moving_average_dyn(&sig, &mut smoothed_sig, 3);
        f.iter_mut().zip(smoothed_sig).for_each(|(pt, y)| {
            *pt.2 = y;
        });
    });

    frame.features = frame.features.as_mut().map(|fm| {
        fm.iter().flat_map(|f| f.split_sparse(0.1)).map(|f| {
            f.to_owned()
        }).collect()
    });

    // mzsignal::text::write_feature_table("raw_features.txt", frame.features.as_ref().unwrap().iter())?;
    let mut deconv = FeatureProcessor::new(
        frame.features.clone().unwrap(),
        CachingIsotopicModel::from(IsotopicModels::Glycopeptide),
        PenalizedMSDeconvScorer::new(0.04, 2.0),
        MaximizingFitFilter::new(1.0),
        2,
        0.02,
        5.0,
        true,
    );

    let params = FeatureSearchParams {
        truncate_after: 0.95,
        max_missed_peaks: 2,
        threshold_scale: 0.3,
        detection_threshold: 0.1,
    };

    let deconv_map = deconv
        .deconvolve(Tolerance::PPM(20.0), (1, 8), 1, 1, &params, 1e-3, 10)
        .unwrap();

    // mzsignal::text::write_feature_table(
    //     "processed_features.txt",
    //     deconv_map.iter().map(|s| s.as_inner().as_inner().0),
    // )?;

    let features_at = deconv_map.all_features_for(3602.55817059969, Tolerance::PPM(10.0));
    let charges: Vec<_> = features_at.iter().map(|f| f.charge()).collect();
    assert!(charges.contains(&3));

    // #[cfg(feature = "serde")]
    // {
    //     serde_json::to_writer_pretty(std::fs::File::create("../processed_features.json").unwrap(), &deconv_map).unwrap();
    // }

    Ok(())
}

#[test]
fn test_map() -> io::Result<()> {
    init_logging();

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
        3,
        0.25,
        5.0,
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
    assert_eq!(features.len(), 4);

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
        debug!(
            "{} solved features, {} hits for query",
            feature_map.len(),
            hits.len()
        );
        for hit in hits {
            debug!(
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
                let envelope: Vec<_> = hit.envelope().iter().map(|e| e.at(i).unwrap()).collect();
                debug!("\t{i}\t{mass:0.3}@{time:0.2} => {inten:0.2} ({envelope:?})",);
            }
        }
    }

    query(4368.263, &deconv_map);
    Ok(())
}
