use std::env;
use std::fs;
use std::io;
use std::path;
use std::thread;
use std::sync::mpsc::{Sender, Receiver, channel};

use log;
use env_logger;

use chemical_elements::PROTON;
use itertools::Itertools;
use mzdata::io::mzml::{MzMLReaderType, MzMLWriterType};
use mzdata::spectrum::SignalContinuity;
use mzdata::prelude::*;
use mzdata::io::traits::ScanWriter;
use mzdeisotope::api::{DeconvolutionEngine, PeaksAndTargets};

use mzdeisotope::isotopic_model::{IsotopicModel, IsotopicModels, IsotopicPatternParams};
use mzdeisotope::scorer::{MSDeconvScorer, MaximizingFitFilter, PenalizedMSDeconvScorer};
use mzdeisotope::solution::DeconvolvedSolutionPeak;
use mzpeaks::CentroidPeak;
use mzpeaks::Tolerance;


fn run_deconvolution(mut reader: MzMLReaderType<fs::File, CentroidPeak, DeconvolvedSolutionPeak>, sender: Sender<SpectrumGroup<CentroidPeak, DeconvolvedSolutionPeak>>) -> io::Result<()> {
    let model: IsotopicModel = IsotopicModels::Peptide.into();

    let mut ms1_engine = DeconvolutionEngine::new(
        Default::default(),
        model.clone().into(),
        PenalizedMSDeconvScorer::new(0.02, 2.0),
        MaximizingFitFilter::new(10.0),
    );

    let mut msn_engine = DeconvolutionEngine::new(
        IsotopicPatternParams::new(0.8, 0.001, None, PROTON),
        model.into(),
        MSDeconvScorer::default(),
        MaximizingFitFilter::new(2.0),
    );

    reader
    .groups()
    .map(|mut group| {
        let precursor_mz: Vec<_> = group
            .products()
            .into_iter()
            .flat_map(|s| s.precursor().and_then(|prec| Some(prec.ion().mz)))
            .collect();
        let targets = match group.precursor_mut() {
            Some(scan) => {
                log::info!("Processing {} MS{} ({:0.3})", scan.id(), scan.ms_level(), scan.acquisition().start_time());
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
                let PeaksAndTargets {
                    deconvoluted_peaks,
                    targets,
                } = ms1_engine.deconvolute_peaks_with_targets(peaks.clone(), Tolerance::PPM(10.0), (1, 8), 1, &precursor_mz);
                scan.deconvoluted_peaks = Some(deconvoluted_peaks);
                targets
            }
            None => precursor_mz.iter().map(|_| None).collect(),
        };
        group.products_mut().iter_mut().for_each(|scan| {
            log::info!("Processing {} MS{} ({:0.3})", scan.id(), scan.ms_level(), scan.acquisition().start_time());
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
            let deconvoluted_peaks = msn_engine.deconvolute_peaks(peaks.clone(), Tolerance::PPM(10.0), (1, 8), 1);
            scan.deconvoluted_peaks = Some(deconvoluted_peaks);
            scan.precursor_mut().and_then(|prec| {
                let target_mz = prec.mz();
                let _ = precursor_mz
                    .iter()
                    .find_position(|t| ((**t) - target_mz).abs() < 1e-6)
                    .and_then(|(i, _)| {
                        if let Some(peak) = &targets[i] {
                            let orig_mz = prec.ion.mz;
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
                                log::warn!("Expected ion of charge state {} @ {orig_mz:0.3}, found {} @ {:0.3}", orig_charge.unwrap(), peak.charge, peak.mz());
                            }
                        };
                        Some(())
                    });

                Some(prec)
            });
        });
        group
    }).for_each(|group|{
        match sender.send(group) {
            Ok(_) => {},
            Err(e) => {
                log::warn!("Failed to send group: {}", e);
            },
        }
    });
    Ok(())
}


fn write_output<W: io::Write + io::Seek>(mut writer: MzMLWriterType<W, CentroidPeak, DeconvolvedSolutionPeak>, receiver: Receiver<SpectrumGroup<CentroidPeak, DeconvolvedSolutionPeak>>) -> io::Result<()> {
    while let Ok(group) = receiver.recv() {
        writer.write_group(&group)?;
    }
    match writer.close() {
        Ok(_) => {},
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

    let reader =
        MzMLReaderType::<fs::File, CentroidPeak, DeconvolvedSolutionPeak>::open_path(inpath.clone())?;

    let outpath = inpath.with_extension("out.mzML");
    let mut writer = MzMLWriterType::new(io::BufWriter::new(fs::File::create(outpath)?));
    writer.copy_metadata_from(&reader);
    let (send, recv) = channel();
    let read_task = thread::spawn(move || run_deconvolution(reader, send));
    let write_task = thread::spawn(move || write_output(writer, recv));

    match read_task.join() {
        Ok(o) => {
            o?
        },
        Err(e) => {
            log::warn!("Failed to join reader task: {:?}", e);
        },
    }

    match write_task.join() {
        Ok(o) => {
            o?
        },
        Err(e) => {
            log::warn!("Failed to join writer task: {:?}", e);
        },
    }
    Ok(())
}
