use std::collections::HashMap;
use std::env;
use std::fs;
use std::io;
use std::path;
use std::sync::atomic::AtomicU16;
use std::sync::atomic::Ordering;
use std::sync::mpsc::TryRecvError;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::time::Duration;
use std::time::Instant;

use env_logger;
use log;
use rayon::prelude::*;

use chemical_elements::PROTON;
use itertools::Itertools;
use mzdata::io::mzml::{MzMLReaderType, MzMLWriterType};
#[allow(unused)]
use mzdata::io::traits::ScanWriter;
use mzdata::prelude::*;
use mzdata::spectrum::SignalContinuity;
use mzdata::Param;
use mzdeisotope::api::{DeconvolutionEngine, PeaksAndTargets};

use mzdeisotope::isotopic_model::{IsotopicModel, IsotopicModels, IsotopicPatternParams};
use mzdeisotope::scorer::{MSDeconvScorer, MaximizingFitFilter, PenalizedMSDeconvScorer};
use mzdeisotope::solution::DeconvolvedSolutionPeak;
use mzpeaks::CentroidPeak;
use mzpeaks::PeakCollection;
use mzpeaks::Tolerance;

type SolvedSpectrumGroup = SpectrumGroup<CentroidPeak, DeconvolvedSolutionPeak>;

#[derive(Debug, Default)]
pub struct SpectrumGroupCollator {
    pub waiting: HashMap<usize, SolvedSpectrumGroup>,
    pub next_key: usize,
    pub ticks: usize,
    pub done: bool,
}

impl SpectrumGroupCollator {
    pub fn receive(&mut self, group_idx: usize, group: SolvedSpectrumGroup) {
        self.waiting.insert(group_idx, group);
    }

    pub fn receive_from(
        &mut self,
        receiver: &Receiver<(usize, SolvedSpectrumGroup)>,
        batch_size: usize,
    ) {
        let mut counter = 0usize;
        while let Ok((group_idx, group)) = receiver.recv_timeout(Duration::from_micros(1)) {
            self.receive(group_idx, group);
            counter += 1;
            if counter > batch_size {
                break;
            }
        }
    }

    pub fn has_next(&self) -> bool {
        self.waiting.contains_key(&self.next_key)
    }

    pub fn try_next(&mut self) -> Option<(usize, SolvedSpectrumGroup)> {
        self.waiting.remove_entry(&self.next_key).and_then(|op| {
            self.next_key += 1;
            Some(op)
        })
    }
}

fn run_deconvolution(
    mut reader: MzMLReaderType<fs::File, CentroidPeak, DeconvolvedSolutionPeak>,
    sender: Sender<(usize, SpectrumGroup<CentroidPeak, DeconvolvedSolutionPeak>)>,
) -> io::Result<()> {
    let model: IsotopicModel = IsotopicModels::Peptide.into();

    let init_counter = AtomicU16::new(0);
    let started = Instant::now();

    let mut ms1_engine = DeconvolutionEngine::new(
        Default::default(),
        model.clone().into(),
        PenalizedMSDeconvScorer::new(0.02, 2.0),
        MaximizingFitFilter::new(10.0),
    );

    let populate_ms1_cache = thread::spawn(move || {
        ms1_engine.populate_isotopic_model_cache(80.0, 3000.0, 1, 8);
        ms1_engine
    });

    let mut msn_engine = DeconvolutionEngine::new(
        IsotopicPatternParams::new(0.8, 0.001, None, PROTON),
        model.into(),
        MSDeconvScorer::default(),
        MaximizingFitFilter::new(2.0),
    );

    let populate_msn_cache = thread::spawn(move || {
        msn_engine.populate_isotopic_model_cache(80.0, 3000.0, 1, 8);
        msn_engine
    });

    ms1_engine = populate_ms1_cache.join().unwrap();
    msn_engine = populate_msn_cache.join().unwrap();

    let (n_ms1_peaks, n_msn_peaks) = reader
        .groups()
        .enumerate()
        .par_bridge()
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
                        // log::info!("Processing {} MS{} ({:0.3})", scan.id(), scan.ms_level(), scan.acquisition().start_time());

                        #[cfg(feature = "verbose")]
                        let logfile =
                            fs::File::create(format!("scan-{}-log.txt", scan.index())).unwrap();

                        let peaks = match scan.signal_continuity() {
                            SignalContinuity::Unknown => {
                                panic!("Can't infer peak mode for {}", scan.id())
                            }
                            SignalContinuity::Centroid => scan.try_build_centroids().unwrap(),
                            SignalContinuity::Profile => {
                                scan.pick_peaks(1.0, Default::default()).unwrap();
                                scan.peaks.as_ref().unwrap()
                            }
                        };

                        #[cfg(feature = "verbose")]
                        ms1_engine.set_log_file(Some(logfile));

                        let PeaksAndTargets {
                            deconvoluted_peaks,
                            targets,
                        } = ms1_engine.deconvolute_peaks_with_targets(
                            peaks.clone(),
                            Tolerance::PPM(20.0),
                            (1, 8),
                            1,
                            &precursor_mz,
                        );
                        n_ms1_peaks = deconvoluted_peaks.len();
                        scan.deconvoluted_peaks = Some(deconvoluted_peaks);
                        targets
                    }
                    None => precursor_mz.iter().map(|_| None).collect(),
                };

                group.products_mut().iter_mut().for_each(|scan| {
                    if !had_precursor {
                        log::info!(
                            "Processing {} MS{} ({:0.3})",
                            scan.id(),
                            scan.ms_level(),
                            scan.acquisition().start_time()
                        );
                    }

                    #[cfg(feature = "verbose")]
                    let logfile =
                        fs::File::create(format!("scan-ms2-{}-log.txt", scan.index())).unwrap();

                    let peaks = match scan.signal_continuity() {
                        SignalContinuity::Unknown => {
                            panic!("Can't infer peak mode for {}", scan.id())
                        }
                        SignalContinuity::Centroid => scan.try_build_centroids().unwrap(),
                        SignalContinuity::Profile => {
                            scan.pick_peaks(1.0, Default::default()).unwrap();
                            scan.peaks.as_ref().unwrap()
                        }
                    };

                    #[cfg(feature = "verbose")]
                    msn_engine.set_log_file(Some(logfile));

                    let deconvoluted_peaks = msn_engine.deconvolute_peaks(
                        peaks.clone(),
                        Tolerance::PPM(20.0),
                        (1, 8),
                        1,
                    );
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
                                    let orig_charge = prec.ion.charge;
                                    let update_ion = if let Some(orig_z) = orig_charge {
                                        orig_z == peak.charge
                                    } else {
                                        true
                                    };
                                    if update_ion {
                                        prec.ion.mz = peak.mz();
                                        prec.ion.charge = Some(peak.charge);
                                        prec.ion.intensity = peak.intensity;
                                    } else {
                                        // log::warn!("Expected ion of charge state {} @ {orig_mz:0.3}, found {} @ {:0.3}", orig_charge.unwrap(), peak.charge, peak.mz());
                                        prec.ion.params_mut().push(Param::new_key_value(
                                            "mzdeisotope:defaulted".to_string(),
                                            true.to_string(),
                                        ));
                                    }
                                };
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
                    log::warn!("Failed to send group: {}", e);
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
    log::debug!(
        "{} threads run for deconvolution",
        init_counter.load(Ordering::SeqCst)
    );
    log::info!("MS1 Peaks: {n_ms1_peaks}\tMSn Peaks: {n_msn_peaks}");
    log::info!("Elapsed Time: {:0.3?}", elapsed);
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
                    log::error!("Failed to send {group_idx} for writing: {e}")
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
        let scan = group.precursor().or_else(|| {
            group.products().into_iter().min_by(|a, b| {
                a
                    .start_time()
                    .total_cmp(&b.start_time())
            })
        }).unwrap();
        scan_counter += group.total_spectra();
        let scan_time = scan.start_time();
        if ((group_idx - checkpoint) % 100 == 0 && group_idx != 0) || (scan_time - time_checkpoint) > 1.0 {
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

fn main() -> io::Result<()> {
    env_logger::init();
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