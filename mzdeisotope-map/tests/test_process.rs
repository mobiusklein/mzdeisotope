#[allow(unused)]
use std::{fs, io, io::prelude::*};

use mzdata::spectrum::bindata::{BinaryArrayMap3D, BuildFromArrayMap3D};
use mzdata::{prelude::*, MzMLWriter};

use mzpeaks::feature::Feature;
use mzpeaks::feature_map::FeatureMap;
use mzpeaks::{IonMobility, Mass, Time, Tolerance, MZ};
use mzsignal::feature_mapping::{FeatureExtracter, FeatureExtracterType};

use mzdeisotope::isotopic_model::{CachingIsotopicModel, IsotopicModels};
use mzdeisotope::scorer::{MaximizingFitFilter, PenalizedMSDeconvScorer};
use mzdeisotope_map::{
    solution::DeconvolvedSolutionFeature, FeatureProcessor, FeatureSearchParams,
};
use rayon::prelude::*;
use tracing::debug;

use mzsignal::feature_statistics::FeatureTransform;

macro_rules! assert_is_close {
    ($t1:expr, $t2:expr, $tol:expr, $label:literal) => {
        assert!(
            ($t1 - $t2).abs() < $tol,
            "Observed {} {}, expected {}, difference {}",
            $label,
            $t1,
            $t2,
            $t1 - $t2,
        );
    };
    ($t1:expr, $t2:expr, $tol:expr, $label:literal, $obj:ident) => {
        assert!(
            ($t1 - $t2).abs() < $tol,
            "Observed {} {}, expected {}, difference {} from {:?}",
            $label,
            $t1,
            $t2,
            $t1 - $t2,
            $obj
        );
    };
}

fn prepare_feature_map() -> io::Result<FeatureMap<MZ, Time, Feature<MZ, Time>>> {
    let reader = mzdata::MZReader::open_path("../test/data/batching_test.mzML")?;

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

    let mut features: FeatureMap<MZ, Time, Feature<MZ, Time>> =
        extractor.extract_features(Tolerance::PPM(10.0), 3, 0.25);

    features.par_iter_mut().for_each(|f| {
        f.smooth(1);
    });
    Ok(features)
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
        for (mz, int) in mzs.iter().copied().zip(ints.iter().copied()) {
            writeln!(writer, "{mz},{time},{int}")?
        }
    }
    Ok(())
}

#[test_log::test]
#[test_log(default_log_filter = "debug")]
fn test_map_im() -> io::Result<()> {
    let sid = "merged=42926 frame=9728 scanStart=1 scanEnd=705";

    let mut reader = mzdata::MzMLReader::new_indexed(mzdata::io::RestartableGzDecoder::new(
        io::BufReader::new(fs::File::open(
            "../test/data/20200204_BU_8B8egg_1ug_uL_7charges_60_min_Slot2-11_1_244.mzML.gz",
        )?),
    )).into_frame_source::<Feature<MZ, IonMobility>, DeconvolvedSolutionFeature<IonMobility>>();


    let mut frame= reader.get_frame_by_id(sid).unwrap();

    frame.extract_features_simple(Tolerance::PPM(15.0), 2, 0.01, None)?;
    frame.features = frame.features.map(|mut fmap| {
        let vfmap: Vec<_> = fmap
            .into_par_iter()
            .filter(|f| f.len() > 1)
            .map(|mut f| {
                f.smooth(1);
                f
            })
            .collect();
        fmap = FeatureMap::new(vfmap);
        fmap
    });

    let mut deconv = FeatureProcessor::new(
        frame.features.clone().unwrap(),
        CachingIsotopicModel::from(IsotopicModels::Glycopeptide),
        PenalizedMSDeconvScorer::new(0.04, 2.0),
        MaximizingFitFilter::new(5.0),
        2,
        0.02,
        5.0,
        true,
    );

    let params = FeatureSearchParams {
        truncate_after: 0.95,
        ignore_below: 0.05,
        max_missed_peaks: 2,
        threshold_scale: 0.3,
        detection_threshold: 0.1,
    };

    let deconv_map = deconv
        .deconvolve(Tolerance::PPM(15.0), (1, 8), 1, 1, &params, 1e-3, 10)
        .unwrap();

    let features_at = deconv_map.all_features_for(3602.55817059969, Tolerance::PPM(10.0));
    let charges: Vec<_> = features_at.iter().map(|f| f.charge()).collect();
    assert!(charges.contains(&3));

    frame.deconvoluted_features = Some(deconv_map);

    let mut buffer: Vec<u8> = Vec::new();
    let write_cursor = io::Cursor::new(&mut buffer);

    let mut writer = MzMLWriter::new(write_cursor);
    writer.set_spectrum_count(1);
    writer.write_frame(&frame)?;
    writer.close()?;
    drop(writer);

    let read_cursor = io::Cursor::new(buffer);
    let reader = mzdata::MZReader::open_read_seek(read_cursor)?;
    let mut reader: mzdata::io::Generic3DIonMobilityFrameSource<
        _,
        _,
        _,
        mzpeaks::feature::Feature<MZ, IonMobility>,
        DeconvolvedSolutionFeature<IonMobility>,
    > = mzdata::io::Generic3DIonMobilityFrameSource::new(reader);
    let dup_frame: mzdata::spectrum::MultiLayerIonMobilityFrame<
        _,
        DeconvolvedSolutionFeature<IonMobility>,
    > = reader.get_frame_by_id(sid).unwrap();

    let dup_features = FeatureMap::new(
        DeconvolvedSolutionFeature::try_from_arrays_3d(dup_frame.arrays.as_ref().unwrap()).unwrap(),
    );

    for (fb, fa) in dup_features
        .iter()
        .zip(frame.deconvoluted_features.as_ref().unwrap().iter())
    {
        let envelope_a = fa.envelope();
        let envelope_b = fb.envelope();
        debug!(
            "Comparing {:0.3}@{:0.3}/{} to {:0.3}@{:0.3}/{}",
            fa.neutral_mass(),
            fa.start_time().unwrap(),
            fa.len(),
            fb.neutral_mass(),
            fb.start_time().unwrap(),
            fb.len()
        );
        assert_eq!(envelope_a[0].len(), envelope_b[0].len());
        assert_is_close!(fa.neutral_mass(), fb.neutral_mass(), 1e-3, "neutral_mass");
        // assert_eq!(fa.len(), fb.len());
        assert_eq!(fa.charge(), fb.charge());
        assert_is_close!(
            fa.start_time().unwrap(),
            fb.start_time().unwrap(),
            1e-3,
            "start_time"
        );
        assert_is_close!(
            fa.end_time().unwrap(),
            fb.end_time().unwrap(),
            1e-3,
            "end_time"
        );

        // let key = (fa.neutral_mass(), fb.neutral_mass());
        // for (ea, eb) in envelope_a.iter().zip(envelope_b.iter()) {
        //     for (ia, ib) in ea.iter().zip(eb.iter()) {
        //         assert_is_close!(ia.0, ib.0, 1e-3, "envelope_mz", key);
        //         assert_is_close!(ia.1, ib.1, 1e-3, "envelope_im", key);
        //         assert_is_close!(ia.2, ib.2, 1e-3, "envelope_int", key);
        //     }
        // }
    }

    Ok(())
}

#[test_log::test]
#[test_log(default_log_filter = "debug")]
fn test_map_im_ms2() -> io::Result<()> {
    let sid = "merged=42853 frame=9714 scanStart=341 scanEnd=359";

    let mut reader = mzdata::MzMLReader::new_indexed(mzdata::io::RestartableGzDecoder::new(
        io::BufReader::new(fs::File::open(
            "../test/data/20200204_BU_8B8egg_1ug_uL_7charges_60_min_Slot2-11_1_244.mzML.gz",
        )?),
    )).into_frame_source::<Feature<MZ, IonMobility>, DeconvolvedSolutionFeature<IonMobility>>();

    let mut frame= reader.get_frame_by_id(sid).unwrap();

    frame.extract_features_simple(Tolerance::PPM(15.0), 1, 0.2, None)?;
    frame.features = frame.features.map(|mut fmap| {
        let vfmap: Vec<_> = fmap
            .into_par_iter()
            .map(|mut f| {
                f.smooth(1);
                f
            })
            .collect();
        fmap = FeatureMap::new(vfmap);
        fmap
    });

    let mut deconv = FeatureProcessor::new(
        frame.features.clone().unwrap(),
        CachingIsotopicModel::from(IsotopicModels::Glycopeptide),
        PenalizedMSDeconvScorer::new(0.04, 2.0),
        MaximizingFitFilter::new(5.0),
        1,
        0.02,
        5.0,
        true,
    );

    let params = FeatureSearchParams {
        truncate_after: 0.8,
        ignore_below: 0.01,
        max_missed_peaks: 2,
        threshold_scale: 0.3,
        detection_threshold: 0.1,
    };

    let deconv_map = deconv
        .deconvolve(Tolerance::PPM(15.0), (1, 8), 1, 1, &params, 1e-3, 10)
        .unwrap();

    let features = deconv_map.all_features_for(203.079, Tolerance::Da(0.02));
    assert_eq!(features.len(), 1);
    assert_eq!(features[0].charge(), 1,);

    frame.deconvoluted_features = Some(deconv_map);

    Ok(())
}

#[test_log::test]
#[test_log(default_log_filter = "debug")]
fn test_map_rt() -> io::Result<()> {
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
        ignore_below: 0.05,
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
            .iter()
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
