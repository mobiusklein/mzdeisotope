use std::{io, sync::mpsc::{Receiver, Sender, TryRecvError}};

use mzdata::prelude::*;

use crate::types::{CPeak, DPeak, SpectrumGroupType, SpectrumGroupCollator};


pub fn collate_results(
    receiver: Receiver<(usize, SpectrumGroupType)>,
    sender: Sender<(usize, SpectrumGroupType)>,
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


pub fn write_output<S: ScanWriter<'static, CPeak, DPeak>>(
    mut writer: S,
    receiver: Receiver<(usize, SpectrumGroupType)>,
) -> io::Result<()> {
    let mut checkpoint = 0usize;
    let mut time_checkpoint = 0.0;
    let mut scan_counter = 0usize;
    let mut scan_time = 0.0;
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
        scan_time = scan.start_time();
        if ((group_idx - checkpoint) % 100 == 0 && group_idx != 0)
            || (scan_time - time_checkpoint) > 1.0
        {
            log::info!("Completed Group {group_idx} | Scans={scan_counter} Time={scan_time:0.3}");
            checkpoint = group_idx;
            time_checkpoint = scan_time;
        }
        writer.write_group(&group)?;
    }
    if time_checkpoint != scan_time {
        log::info!("Finished Processing | Scans={scan_counter} Time={scan_time:0.3}");
    }
    writer.close()?;
    // match writer.close() {
    //     Ok(_) => {}
    //     Err(e) => match e {
    //         mzdata::MzMLWriterError::IOError(o) => return Err(o),
    //         _ => Err(io::Error::new(io::ErrorKind::InvalidInput, e))?,
    //     },
    // };
    Ok(())
}
