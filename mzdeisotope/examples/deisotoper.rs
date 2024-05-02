use std::env;
use std::fs;
use std::io;
use std::path;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Instant;

use rayon::prelude::*;

use chemical_elements::PROTON;
use itertools::Itertools;
use mzdata::io::{
    mzml::{MzMLReaderType, MzMLWriterType},
    SpectrumWriter,
};
#[allow(unused)]
use mzdata::prelude::*;
use mzdata::spectrum::{utils::Collator, SignalContinuity, SpectrumGroup};
use mzdata::Param;

use mzdeisotope::api::{DeconvolutionEngine, PeaksAndTargets};
use mzdeisotope::isotopic_model::{IsotopicModel, IsotopicModels, IsotopicPatternParams};
use mzdeisotope::scorer::{MSDeconvScorer, MaximizingFitFilter, PenalizedMSDeconvScorer};
use mzdeisotope::solution::DeconvolvedSolutionPeak;

use mzpeaks::{CentroidPeak, PeakCollection, Tolerance};

use tracing_subscriber::{fmt, prelude::*, EnvFilter};

type SolvedSpectrumGroup = SpectrumGroup<CentroidPeak, DeconvolvedSolutionPeak>;

type SpectrumGroupCollator = Collator<SolvedSpectrumGroup>;

fn run_deconvolution(
    mut reader: MzMLReaderType<fs::File, CentroidPeak, DeconvolvedSolutionPeak>,
    sender: Sender<(usize, SpectrumGroup<CentroidPeak, DeconvolvedSolutionPeak>)>,
) -> io::Result<()> {
    let model: IsotopicModel = IsotopicModels::Peptide.into();

    let init_counter = AtomicU16::new(0);
    let started = Instant::now();

    let mut ms1_engine = DeconvolutionEngine::new(
        Default::default(),
        model.clone(),
        PenalizedMSDeconvScorer::new(0.02, 2.0),
        MaximizingFitFilter::new(10.0),
        true,
    );

    let populate_ms1_cache = thread::spawn(move || {
        ms1_engine.populate_isotopic_model_cache(80.0, 3000.0, 1, 8);
        ms1_engine
    });

    let mut msn_engine = DeconvolutionEngine::new(
        IsotopicPatternParams::new(0.8, 0.001, None, PROTON),
        model.clone(),
        MSDeconvScorer::default(),
        MaximizingFitFilter::new(2.0),
        true,
    );

    let populate_msn_cache = thread::spawn(move || {
        msn_engine.populate_isotopic_model_cache(80.0, 3000.0, 1, 8);
        msn_engine
    });

    ms1_engine = populate_ms1_cache.join().unwrap();
    msn_engine = populate_msn_cache.join().unwrap();

    let (grouper, averager, reprofiler) =
        reader.groups().averaging_deferred(1, 120.0, 2000.1, 0.005);

    let (n_ms1_peaks, n_msn_peaks) = grouper
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
            |(ms1_engine, msn_engine), (group_idx, mut group)| {
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
                        // tracing::info!("Processing {} MS{} ({:0.3})", scan.id(), scan.ms_level(), scan.acquisition().start_time());

                        let peaks = match scan.signal_continuity() {
                            SignalContinuity::Unknown => {
                                panic!("Can't infer peak mode for {}", scan.id())
                            }
                            SignalContinuity::Centroid => scan.try_build_centroids().unwrap(),
                            SignalContinuity::Profile => {
                                scan.pick_peaks(1.0).unwrap();
                                scan.description_mut().signal_continuity =
                                    SignalContinuity::Centroid;
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
                                (1, 8),
                                1,
                                &precursor_mz,
                            )
                            .unwrap();
                        n_ms1_peaks = deconvoluted_peaks.len();
                        scan.deconvoluted_peaks = Some(deconvoluted_peaks);
                        targets
                    }
                    None => precursor_mz.iter().map(|_| None).collect(),
                };

                group.products_mut().iter_mut().for_each(|scan| {
                    if !had_precursor {
                        tracing::info!(
                            "Processing {} MS{} ({:0.3})",
                            scan.id(),
                            scan.ms_level(),
                            scan.acquisition().start_time()
                        );
                    }

                    let peaks = match scan.signal_continuity() {
                        SignalContinuity::Unknown => {
                            panic!("Can't infer peak mode for {}", scan.id())
                        }
                        SignalContinuity::Centroid => scan.try_build_centroids().unwrap(),
                        SignalContinuity::Profile => {
                            scan.pick_peaks(1.0).unwrap();
                            scan.description_mut().signal_continuity = SignalContinuity::Centroid;
                            scan.peaks.as_ref().unwrap()
                        }
                    };

                    let deconvoluted_peaks = msn_engine
                        .deconvolute_peaks(peaks.clone(), Tolerance::PPM(20.0), (1, 8), 1)
                        .unwrap();
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
                                    let orig_charge = prec.ion().charge;
                                    let update_ion = if let Some(orig_z) = orig_charge {
                                        orig_z == peak.charge
                                    } else {
                                        true
                                    };
                                    if update_ion {
                                        prec.ion_mut().mz = peak.mz();
                                        prec.ion_mut().charge = Some(peak.charge);
                                        prec.ion_mut().intensity = peak.intensity;
                                    } else {
                                        // tracing::warn!("Expected ion of charge state {} @ {orig_mz:0.3}, found {} @ {:0.3}", orig_charge.unwrap(), peak.charge, peak.mz());
                                        prec.ion_mut().params_mut().push(Param::new_key_value(
                                            "mzdeisotope:defaulted".to_string(),
                                            true.to_string(),
                                        ));
                                    }
                                };
                                Some(())
                            })
                            .or_else(|| {
                                prec.ion_mut().params_mut().push(Param::new_key_value(
                                    "mzdeisotope:defaulted".to_string(),
                                    true.to_string(),
                                ));
                                prec.ion_mut().params_mut().push(Param::new_key_value(
                                    "mzdeisotope:orphan".to_string(),
                                    true.to_string(),
                                ));
                                None
                            });

                        Some(prec)
                    });
                });
                (group_idx, group, n_ms1_peaks, n_msn_peaks)
            },
        )
        .map(|(group_idx, group, n_ms1_peaks_local, n_msn_peaks_local)| {
            match sender.send((group_idx, group)) {
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!("Failed to send group: {}", e);
                }
            }
            (n_ms1_peaks_local, n_msn_peaks_local)
        })
        .reduce(
            || (0usize, 0usize),
            |state, counts| (state.0 + counts.0, state.1 + counts.1),
        );
    let finished = Instant::now();
    let elapsed = finished - started;
    tracing::debug!(
        "{} threads run for deconvolution",
        init_counter.load(Ordering::SeqCst)
    );
    tracing::info!("MS1 Peaks: {n_ms1_peaks}\tMSn Peaks: {n_msn_peaks}");
    tracing::info!("Elapsed Time: {:0.3?}", elapsed);
    Ok(())
}

fn collate_results(
    receiver: Receiver<(usize, SolvedSpectrumGroup)>,
    sender: Sender<(usize, SolvedSpectrumGroup)>,
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
                    tracing::error!("Failed to send {group_idx} for writing: {e}")
                }
            }
        }
    }
}

fn write_output<W: io::Write + io::Seek>(
    mut writer: MzMLWriterType<W, CentroidPeak, DeconvolvedSolutionPeak>,
    receiver: Receiver<(usize, SolvedSpectrumGroup)>,
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
            tracing::info!(
                "Completed Group {group_idx} | Scans={scan_counter} Time={scan_time:0.3}"
            );
            checkpoint = group_idx;
            time_checkpoint = scan_time;
        }
        writer.write_group(&group)?;
    }
    match writer.close() {
        Ok(_) => {}
        Err(e) => match e {
            mzdata::io::mzml::MzMLWriterError::IOError(o) => return Err(o),
            _ => Err(io::Error::new(io::ErrorKind::InvalidInput, e))?,
        },
    };
    Ok(())
}

fn main() -> io::Result<()> {
    tracing_subscriber::registry()
        .with(
            fmt::layer().compact().with_writer(io::stderr).with_filter(
                EnvFilter::builder()
                    .with_default_directive(tracing::Level::INFO.into())
                    .from_env_lossy(),
            ),
        )
        .init();

    let mut args = env::args().skip(1);
    let inpath = path::PathBuf::from(args.next().unwrap());

    let reader = MzMLReaderType::<fs::File, CentroidPeak, DeconvolvedSolutionPeak>::open_path(
        inpath.clone(),
    )?;

    let outpath = inpath.with_extension("out.mzML");
    let mut writer = MzMLWriterType::new(io::BufWriter::new(fs::File::create(outpath)?));
    writer.copy_metadata_from(&reader);

    let (send_solved, recv_solved) = channel();
    let (send_collated, recv_collated) = channel();

    let read_task = thread::spawn(move || run_deconvolution(reader, send_solved));

    let collate_task = thread::spawn(move || collate_results(recv_solved, send_collated));

    let write_task = thread::spawn(move || write_output(writer, recv_collated));

    match read_task.join() {
        Ok(o) => o?,
        Err(e) => {
            tracing::warn!("Failed to join reader task: {e:?}");
        }
    }

    match collate_task.join() {
        Ok(_) => {}
        Err(e) => {
            tracing::warn!("Failed to join collator task: {e:?}")
        }
    }

    match write_task.join() {
        Ok(o) => o?,
        Err(e) => {
            tracing::warn!("Failed to join writer task: {e:?}");
        }
    }
    Ok(())
}
