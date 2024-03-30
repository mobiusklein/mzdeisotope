use std::{
    io,
    sync::mpsc::{Receiver, SyncSender, TryRecvError},
};

use mzdata::prelude::*;

use crate::types::{
    CPeak, DPeak, SpectrumCollator, SpectrumGroupType, SpectrumType,
};

pub fn collate_results_spectra(
    receiver: Receiver<(usize, SpectrumGroupType)>,
    sender: SyncSender<(usize, SpectrumType)>,
) {
    let mut collator = SpectrumCollator::default();
    let mut i = 0;
    loop {
        i += 1;
        match receiver.try_recv() {
            Ok((group_idx, group)) => {
                if group_idx == 0 {
                    if let Some(i) = group.iter().map(|s| s.index()).min() {
                        collator.next_key = i;
                    }
                }
                group
                    .into_iter()
                    .for_each(|s| collator.receive(s.index(), s))
                // collator.receive_from(&receiver, 100);
            }
            Err(e) => match e {
                TryRecvError::Empty => {}
                TryRecvError::Disconnected => {
                    collator.done = true;
                    break;
                }
            },
        }

        let n = collator.waiting.len();
        if i % 1000000 == 0 && i > 0 && n > 0 {
            tracing::debug!("Collator holding {n} entries at tick {i}, next key {} ({})", collator.next_key, collator.has_next())
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
                tracing::info!("Completed Scan {} | Scans={scan_counter} Time={scan_time:0.3}", group_idx + 1);
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
