use std::{
    io,
    sync::mpsc::{Receiver, SyncSender, TryRecvError, TrySendError}
};

use itertools::Itertools;
use mzdata::{io::MassSpectrometryFormat, prelude::*, spectrum::bindata::BinaryCompressionType};
use tracing::{debug, error, info};
use std::time::Instant;

use crate::types::{CPeak, DPeak, SpectrumCollator, SpectrumGroupType, SpectrumType, BUFFER_SIZE};

pub(crate) fn postprocess_spectra(
    group_idx: usize,
    mut group: SpectrumGroupType,
    output_format: MassSpectrometryFormat,
) -> (usize, SpectrumGroupType) {
    if matches!(output_format, MassSpectrometryFormat::MzML) {
        if let Some(precursor) = group.precursor_mut() {
            if let Some(peaks) = precursor.deconvoluted_peaks.as_ref() {
                let mut arrays = BuildArrayMapFrom::as_arrays(peaks);
                arrays.iter_mut().for_each(|(_k, a)| {
                    a.store_compressed(BinaryCompressionType::Zlib).unwrap();
                });
                precursor.arrays = Some(arrays);
                precursor.deconvoluted_peaks = None;
                precursor.peaks = None
            }
        }
        for product in group.products_mut().iter_mut() {
            if let Some(peaks) = product.deconvoluted_peaks.as_ref() {
                let mut arrays = BuildArrayMapFrom::as_arrays(peaks);
                arrays.iter_mut().for_each(|(_k, a)| {
                    a.store_compressed(BinaryCompressionType::Zlib).unwrap();
                });
                product.arrays = Some(arrays);
                product.deconvoluted_peaks = None;
                product.peaks = None;
            }
        }
        (group_idx, group)
    } else if matches!(output_format, MassSpectrometryFormat::MzMLb) {
        if let Some(precursor) = group.precursor_mut() {
            if let Some(peaks) = precursor.deconvoluted_peaks.as_ref() {
                let arrays = BuildArrayMapFrom::as_arrays(peaks);
                precursor.arrays = Some(arrays);
                precursor.deconvoluted_peaks = None;
                precursor.peaks = None
            }
        }
        for product in group.products_mut().iter_mut() {
            if let Some(peaks) = product.deconvoluted_peaks.as_ref() {
                let arrays = BuildArrayMapFrom::as_arrays(peaks);
                product.arrays = Some(arrays);
                product.deconvoluted_peaks = None;
                product.peaks = None;
            }
        }
        (group_idx, group)
    } else {
        (group_idx, group)
    }
}

fn drain_channel(collator: &mut SpectrumCollator, channel: &Receiver<(usize, SpectrumGroupType)>, batch_size: usize) -> bool {
    for b in 0..batch_size {
        match channel.try_recv() {
            Ok((_, group)) => {
                group
                    .into_iter()
                    .for_each(|s| collator.receive(s.index(), s));
            }
            Err(e) => match e {
                TryRecvError::Empty => {
                    if b > batch_size / 2 {
                        info!("Drained {b} items from work queue");
                    }
                    break;
                }
                TryRecvError::Disconnected => {
                    collator.done = true;
                    break;
                }
            },
        }
    }
    collator.done
}

pub fn collate_results_spectra(
    receiver: Receiver<(usize, SpectrumGroupType)>,
    sender: SyncSender<(usize, SpectrumType)>,
) {
    let mut collator = SpectrumCollator::default();
    let mut i = 0usize;
    let mut last_send = Instant::now();
    let mut has_work = true;
    while has_work {
        i += 1;
        if i == usize::MAX {
            info!("Collation counter rolling over");
            i = 1;
        }
        match receiver.try_recv() {
            Ok((group_idx, group)) => {
                if group_idx == 0 {
                    if let Some(i) = group.iter().map(|s| s.index()).min() {
                        collator.next_key = i;
                    }
                }
                group
                    .into_iter()
                    .for_each(|s| collator.receive(s.index(), s));
                if drain_channel(&mut collator, &receiver, 1000) {
                    if collator.done && collator.waiting.is_empty() {
                        debug!("Setting collator loop condition to false");
                        has_work = false;
                    }
                }
            }
            Err(e) => match e {
                TryRecvError::Empty => {}
                TryRecvError::Disconnected => {
                    collator.done = true;
                    if collator.done && collator.waiting.is_empty() {
                        debug!("Setting collator loop condition to false");
                        has_work = false;
                    }
                }
            },
        }

        let n = collator.waiting.len();
        if !collator.has_next() {
            if i % 10000000 == 0 && i > 0 && n > 0 {
                let t = Instant::now();
                if (t - last_send).as_secs_f64() > 30.0 {
                    let waiting_keys: Vec<_> =
                        collator.waiting.keys().sorted().take(10).copied().collect();
                    tracing::info!(
                       "Collator holding {n} entries at tick {i}, next key {} ({}), pending keys: {waiting_keys:?}",
                        collator.next_key,
                        collator.has_next()
                    );
                    last_send = t;
                }
            }
        }

        if collator.done && n > 0 {
            debug!("Draining output queue, {n} items");
            let mut waiting_items = std::mem::take(&mut collator.waiting).into_iter().collect_vec();
            waiting_items.sort_by(|(i, _), (j, _)| i.cmp(j));
            for (group_idx, group) in waiting_items {
                match sender.send((group_idx, group)) {
                    Ok(()) => {},
                    Err(e) => {
                        error!("Failed to send {group_idx} for writing: {e}");
                        debug!("Setting collator loop condition to false");
                        has_work = false;
                        break;
                    }
                }
            }
        } else {
            while let Some((group_idx, group)) = collator.try_next() {
                if collator.waiting.len() >= BUFFER_SIZE {
                    match sender.send((group_idx, group)) {
                        Ok(()) => {},
                        Err(e) => {
                            error!("Failed to send {group_idx} for writing: {e}");
                            debug!("Setting collator loop condition to false");
                            has_work = false;
                            break;
                        }
                    }
                } else {
                    match sender.try_send((group_idx, group)) {
                        Ok(()) => {}
                        Err(e) => {
                            match e {
                                TrySendError::Full((group_idx, group)) => {
                                    collator.receive(group_idx, group);
                                    collator.set_next_key(group_idx);
                                },
                                TrySendError::Disconnected(_) => {
                                    error!("Failed to send {group_idx} for writing: {e}");
                                    debug!("Setting collator loop condition to false");
                                    has_work = false;
                                    break;
                                },
                            }
                        }
                    }
                }
            }
        }
        if collator.waiting.len() < n {
            last_send = Instant::now();
        }
    }
    debug!("Spectrum collator done");
}

pub fn write_output_spectra<S: SpectrumWriter<CPeak, DPeak>>(
    mut writer: S,
    receiver: Receiver<(usize, SpectrumType)>,
) -> io::Result<()> {
    let mut checkpoint = 0usize;
    let mut time_checkpoint = 0.0;
    let mut scan_counter = 0usize;
    let mut scan_time = 0.0;
    while let Ok((group_idx, scan)) = receiver.recv() {
        scan_counter += 1;
        scan_time = scan.start_time();
        if ((group_idx - checkpoint) % 1000 == 0 && group_idx != 0)
            || (scan_time - time_checkpoint) > 1.0
        {
            if group_idx + 1 != scan_counter {
                tracing::info!(
                    "Completed Scan {} | Scans={scan_counter} Time={scan_time:0.3}",
                    group_idx + 1
                );
            } else {
                tracing::info!("Completed Scan {} | Time={scan_time:0.3}", group_idx + 1);
            }
            checkpoint = group_idx;
            time_checkpoint = scan_time;
        }
        writer.write(&scan)?;
    }
    if time_checkpoint != scan_time {
        tracing::info!("Finished | Scans={scan_counter} Time={scan_time:0.3}");
    }
    writer.close()?;
    Ok(())
}
