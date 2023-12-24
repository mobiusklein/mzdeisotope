use std::env;
use std::fmt::Display;
use std::io;

use env_logger;
use itertools::Itertools;
use log;

use mzdata::io::mzml::MzMLReaderType;
use mzdata::prelude::*;

use mzdeisotope::solution::DeconvolvedSolutionPeak;
use mzpeaks::CentroidPeak;
use mzpeaks::PeakCollection;
use num_traits::Float;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MatchingStatus {
    Matched,
    MismatchingPeakCount(usize, usize),
    MismatchingPeakValues(usize, usize, usize),
}

impl Display for MatchingStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub fn isclose<T: Float>(a: T, b: T, delta: T) -> bool {
    (a - b).abs() < delta
}

fn main() -> io::Result<()> {
    env_logger::init();
    let mut args = env::args().skip(1);

    let path1 = args.next().unwrap_or_else(|| panic!("Missing first file"));
    let path2 = args.next().unwrap_or_else(|| panic!("Missing second file"));

    let reader1 = MzMLReaderType::<_, CentroidPeak, DeconvolvedSolutionPeak>::open_path(path1)?;
    let reader2 = MzMLReaderType::<_, CentroidPeak, DeconvolvedSolutionPeak>::open_path(path2)?;

    let counts = reader1
        .zip(reader2)
        .map(|(mut scan1, mut scan2)| {
            scan1.try_build_deconvoluted_centroids().unwrap();
            scan2.try_build_deconvoluted_centroids().unwrap();

            let dpeaks1 = scan1.deconvoluted_peaks.as_ref().unwrap();
            let dpeaks2 = scan2.deconvoluted_peaks.as_ref().unwrap();
            let min_score1 = dpeaks1
                .iter()
                .map(|p| p.score)
                .min_by(|a, b| a.total_cmp(b))
                .unwrap();
            assert!(min_score1 >= 2.0, "Minimum score {}", min_score1);
            if dpeaks1.len() == dpeaks2.len() {
                let (
                    mismatched_masses,
                    mismatched_intensities,
                    mismatched_scores,
                    intensity_deltas,
                ) = dpeaks1.iter().zip(dpeaks2.iter()).fold(
                    (0usize, 0usize, 0usize, Vec::new()),
                    |(
                        mut mismatched_masses,
                        mut mismatched_intensities,
                        mut mismatched_scores,
                        mut intensity_deltas,
                    ),
                     (p1, p2)| {
                        if !isclose(p1.neutral_mass, p2.neutral_mass, 1e-3) {
                            mismatched_masses += 1;
                        }
                        if !isclose(p1.intensity, p2.intensity, 1e-3) {
                            mismatched_intensities += 1;
                            intensity_deltas.push((
                                p1.neutral_mass,
                                p1.intensity - p2.intensity,
                                (p1.intensity - p2.intensity) / p1.intensity,
                                p1.score - p2.score, (p1.score - p2.score) / p1.score
                            ));
                        } else if !isclose(p1.score, p2.score, 1e-3) {
                            mismatched_scores += 1;
                            intensity_deltas.push((
                                p1.neutral_mass,
                                p1.intensity - p2.intensity,
                                (p1.intensity - p2.intensity) / p1.intensity,
                                p1.score - p2.score, (p1.score - p2.score) / p1.score
                            ));
                        }

                        (
                            mismatched_masses,
                            mismatched_intensities,
                            mismatched_scores,
                            intensity_deltas,
                        )
                    },
                );
                if mismatched_intensities == 0 && mismatched_masses == 0 && mismatched_scores == 0 {
                    (MatchingStatus::Matched, scan1.ms_level())
                } else {
                    log::info!(
                        "Mismatching peak values {} {}/{}/{} {:?}",
                        scan1.id(),
                        mismatched_masses,
                        mismatched_intensities,
                        mismatched_scores,
                        intensity_deltas.as_slice()
                    );
                    (
                        MatchingStatus::MismatchingPeakValues(
                            mismatched_masses,
                            mismatched_intensities,
                            mismatched_scores,
                        ),
                        scan1.ms_level(),
                    )
                }
            } else {
                log::info!("Mismatching peak count {}", scan1.id());
                (
                    MatchingStatus::MismatchingPeakCount(dpeaks1.len(), dpeaks2.len()),
                    scan1.ms_level(),
                )
            }
        })
        .counts();

    let pairs: Vec<_> = counts.iter().sorted_by_key(|x| (*x.1, *x.0)).collect();
    pairs.iter().for_each(|((outcome, ms_level), count)| {
        println!("{}:{} -> {}", outcome, ms_level, count);
    });
    Ok(())
}
