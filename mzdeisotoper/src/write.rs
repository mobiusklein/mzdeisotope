use std::io;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError};

use itertools::Itertools;
use mzdata::{
    io::MassSpectrometryFormat,
    prelude::*,
    spectrum::{
        bindata::BinaryCompressionType, utils::Collator,
        IonMobilityFrameGroup, SpectrumLike,
    },
    RawSpectrum,
};
use tracing::{debug, error, info, instrument};

use crate::{
    selection_targets::TargetTrackingFrameGroup,
    types::{
        CFeature, CPeak, DFeature, DPeak, FrameGroupType, FrameResult, FrameResultGroup, FrameType, SpectrumGroupType, SpectrumType, BUFFER_SIZE
    },
};

pub(crate) trait HasIndex {
    fn index(&self) -> usize;
}

pub(crate) trait Iterable: IntoIterator {
    fn iter(&self) -> impl Iterator<Item = &Self::Item>;
}

impl Iterable for SpectrumGroupType {
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.iter()
    }
}

impl HasIndex for SpectrumType {
    fn index(&self) -> usize {
        self.description().index
    }
}

impl Iterable for FrameGroupType {
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.iter()
    }
}

impl Iterable for TargetTrackingFrameGroup<CFeature, DFeature, FrameGroupType> {
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.group.iter()
    }
}

impl Iterable for FrameResultGroup {
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.iter()
    }
}

impl HasIndex for FrameType {
    fn index(&self) -> usize {
        self.description().index
    }
}

impl HasIndex for RawSpectrum {
    fn index(&self) -> usize {
        self.description().index
    }
}

impl HasIndex for FrameResult {
    fn index(&self) -> usize {
        IonMobilityFrameLike::index(self)
    }
}

#[instrument(level = "debug", skip(group, output_format))]
pub(crate) fn postprocess_spectra(
    group_idx: usize,
    mut group: SpectrumGroupType,
    output_format: MassSpectrometryFormat,
) -> (usize, SpectrumGroupType) {
    if matches!(output_format, MassSpectrometryFormat::MzML) {
        if let Some(precursor) = group.precursor_mut() {
            if let Some(peaks) = precursor.deconvoluted_peaks.as_ref() {
                let mut arrays = BuildArrayMapFrom::as_arrays(peaks.as_slice());
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
                let mut arrays = BuildArrayMapFrom::as_arrays(peaks.as_slice());
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
                let arrays = BuildArrayMapFrom::as_arrays(peaks.as_slice());
                precursor.arrays = Some(arrays);
                precursor.deconvoluted_peaks = None;
                precursor.peaks = None
            }
        }
        for product in group.products_mut().iter_mut() {
            if let Some(peaks) = product.deconvoluted_peaks.as_ref() {
                let arrays = BuildArrayMapFrom::as_arrays(peaks.as_slice());
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

#[instrument(level = "debug", skip(group, output_format))]
pub(crate) fn postprocess_frames(
    group_idx: usize,
    mut group: TargetTrackingFrameGroup<CFeature, DFeature, FrameGroupType>,
    output_format: MassSpectrometryFormat,
) -> (
    usize,
    FrameResultGroup,
) {
    let mut grp = IonMobilityFrameGroup::default();
    if matches!(output_format, MassSpectrometryFormat::MzML) {
        if let Some(precursor) = group.group.precursor.take() {
            let descr = precursor.description().clone();
            let mut spec = RawSpectrum::from(precursor);
            spec.arrays.iter_mut().for_each(|(_, a)| { a.store_compressed(BinaryCompressionType::Zlib).unwrap() });
            let spec = FrameResult::Flattened(spec, descr);
            grp.precursor = Some(spec);
        }
        let mut products = Vec::new();
        for product in group.group.products {
            let descr = product.description().clone();
            let mut spec = RawSpectrum::from(product);
            spec.arrays.iter_mut().for_each(|(_, a)| { a.store_compressed(BinaryCompressionType::Zlib).unwrap() });
            let spec = FrameResult::Flattened(spec, descr);
            products.push(spec);
        }
        grp.products = products;
        (group_idx, grp)
    } else if matches!(output_format, MassSpectrometryFormat::MzMLb) {
        let mut grp = IonMobilityFrameGroup::default();
        if let Some(precursor) = group.group.precursor.take() {
            let descr = precursor.description().clone();
            let spec = RawSpectrum::from(precursor);
            let spec = FrameResult::Flattened(spec, descr);
            grp.precursor = Some(spec);
        }
        let mut products = Vec::new();
        for product in group.group.products {
            let descr = product.description().clone();
            let spec = FrameResult::Flattened(RawSpectrum::from(product), descr);
            products.push(spec);
        }
        grp.products = products;
        (group_idx, grp)
    } else {
        grp.precursor = group.group.precursor.take().map(FrameResult::Frame);
        grp.products = group
            .group
            .products
            .into_iter()
            .map(FrameResult::Frame)
            .collect();
        (group_idx, grp)
    }
}

fn drain_channel<T: Send + HasIndex, I: Iterable<Item = T>>(
    collator: &mut Collator<T>,
    channel: &Receiver<(usize, I)>,
    batch_size: usize,
) -> bool {
    let mut j = 0;
    for b in 0..batch_size {
        match channel.try_recv() {
            Ok((_, group)) => {
                group
                    .into_iter()
                    .for_each(|s| collator.receive(s.index(), s));
                j += 1;
            }
            Err(e) => match e {
                TryRecvError::Empty => {
                    break;
                }
                TryRecvError::Disconnected => {
                    debug!("Work queue finished after draining {b} items from the work queue");
                    collator.done = true;
                    break;
                }
            },
        }
    }
    if j > batch_size / 2 {
        info!("Drained {j} items from work queue");
    }
    collator.done
}

pub(crate) fn collate_results_spectra<T: HasIndex + Send, I: Iterable<Item = T>>(
    receiver: Receiver<(usize, I)>,
    sender: Sender<(usize, T)>,
) {
    let mut collator = Collator::default();
    let mut i = 0usize;
    let mut last_send = Instant::now();
    let mut has_work = true;
    while has_work {
        i += 1;
        if i == usize::MAX {
            info!("Collation counter rolling over");
            i = 1;
        }
        {
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
                    if drain_channel(&mut collator, &receiver, 1000)
                        && collator.done
                        && collator.waiting.is_empty()
                    {
                        debug!("Setting collator loop condition to false");
                        has_work = false;
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
        }

        let n = collator.waiting.len();
        if !collator.has_next() && i % 10000000 == 0 && i > 0 && n > 0 {
            let t = Instant::now();
            if (t - last_send).as_secs_f64() > 30.0 {
                let waiting_keys: Vec<_> =
                    collator.waiting.keys().sorted().take(10).copied().collect();
                let write_queue_size = sender.len();
                let work_queue_size = receiver.len();
                tracing::info!(
                    r#"Collator holding {n} entries at tick {i}, next key {} ({}), pending keys: {waiting_keys:?}
with {write_queue_size} writing backlog and {work_queue_size} work waiting to collate"#,
                    collator.next_key,
                    collator.has_next()
                );
                last_send = t;
            }
        }

        if collator.done && n > 0 {
            debug!("Draining output queue, {n} items");
            let mut waiting_items = std::mem::take(&mut collator.waiting)
                .into_iter()
                .collect_vec();
            waiting_items.sort_by(|(i, _), (j, _)| i.cmp(j));
            for (group_idx, group) in waiting_items {
                match sender.send((group_idx, group)) {
                    Ok(()) => {}
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
                        Ok(()) => {}
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
                        Err(e) => match e {
                            TrySendError::Full((group_idx, group)) => {
                                collator.receive(group_idx, group);
                                collator.set_next_key(group_idx);
                            }
                            TrySendError::Disconnected(_) => {
                                error!("Failed to send {group_idx} for writing: {e}");
                                debug!("Setting collator loop condition to false");
                                has_work = false;
                                break;
                            }
                        },
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
            let queue_size = receiver.len();
            if group_idx + 1 != scan_counter {
                tracing::info!(
                    "Completed Scan {} | Scans={scan_counter} Time={scan_time:0.3} | {queue_size} items in the write queue",
                    group_idx + 1
                );
            } else {
                tracing::info!("Completed Scan {} | Time={scan_time:0.3} | {queue_size} items in the write queue", group_idx + 1);
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

pub fn write_output_frames<
    S: IonMobilityFrameWriter<CFeature, DFeature> + SpectrumWriter<CPeak, DPeak>,
>(
    mut writer: S,
    receiver: Receiver<(usize, FrameResult)>,
) -> io::Result<()> {
    let mut checkpoint = 0usize;
    let mut time_checkpoint = 0.0;
    let mut frame_counter = 0usize;
    let mut frame_time = 0.0;
    while let Ok((group_idx, frame)) = receiver.recv() {
        frame_counter += 1;
        frame_time = frame.start_time();
        if ((group_idx - checkpoint) % 1000 == 0 && group_idx != 0)
            || (frame_time - time_checkpoint) > 1.0
        {
            let queue_size = receiver.len();
            if group_idx + 1 != frame_counter {
                tracing::info!(
                    "Completed Frame {} | Frames={frame_counter} Time={frame_time:0.3} | {queue_size} items in the write queue",
                    group_idx + 1
                );
            } else {
                tracing::info!("Completed Frame {} | Time={frame_time:0.3} | {queue_size} items in the write queue", group_idx + 1);
            }
            checkpoint = group_idx;
            time_checkpoint = frame_time;
        }
        let span = tracing::debug_span!("writing frame", frame_index = group_idx);
        let _entered = span.enter();

        match frame {
            FrameResult::Flattened(raw_spectrum, _) => writer.write_owned(raw_spectrum),
            FrameResult::Frame(frame) => writer.write_frame_owned(frame),
        }?;

        // writer.write_frame(&frame)?;
    }
    if time_checkpoint != frame_time {
        tracing::info!("Finished | Frame={frame_counter} Time={frame_time:0.3}");
    }
    writer.close()?;
    Ok(())
}
