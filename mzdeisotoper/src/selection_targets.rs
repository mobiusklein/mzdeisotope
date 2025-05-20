use std::cmp::Ordering;
use std::collections::VecDeque;
use std::marker::PhantomData;

use mzdata::prelude::*;
use mzdata::spectrum::{
    IonMobilityFrameGroup, IonMobilityFrameGroupIntoIter, IonMobilityFrameGroupIter, MultiLayerIonMobilityFrame, MultiLayerSpectrum, SpectrumGroup, SpectrumGroupIntoIter, SpectrumGroupIter
};
use mzpeaks::coordinate::{SimpleInterval, Span1D};
use mzpeaks::{CentroidLike, DeconvolutedCentroidLike, IonMobility, Mass, Tolerance, MZ};

use crate::time_range::TimeRange;
use crate::types::{CPeak, DPeak};

pub trait SpectrumGroupTiming {
    fn earliest_time(&self) -> Option<f64>;
    #[allow(unused)]
    fn latest_time(&self) -> Option<f64>;
}

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct SelectionTargetSpecification {
    pub mz: f64,
    pub charge: Option<i32>,
    pub time_range: TimeRange,
}

impl SelectionTargetSpecification {
    pub fn new(mz: f64, charge: Option<i32>, time_range: TimeRange) -> Self {
        Self {
            mz,
            charge,
            time_range,
        }
    }

    pub fn spans(&self, time: f64) -> bool {
        self.time_range.start <= time && self.time_range.end > time
    }
}

impl PartialOrd for SelectionTargetSpecification {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.mz.partial_cmp(&other.mz) {
            Some(Ordering::Equal) => {}
            ord => return ord,
        }
        match self.charge.partial_cmp(&other.charge) {
            Some(Ordering::Equal) => {}
            ord => return ord,
        }
        match self.time_range.start.partial_cmp(&other.time_range.start) {
            Some(Ordering::Equal) => {}
            ord => return ord,
        }
        self.time_range.end.partial_cmp(&other.time_range.end)
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct SelectedTarget {
    pub mz: f64,
    pub charge: Option<i32>,
}

impl SelectedTarget {
    pub fn new(mz: f64, charge: Option<i32>) -> Self {
        Self { mz, charge }
    }
}

#[derive(Debug, Default)]
pub struct TargetTrackingSpectrumGroup<
    C: CentroidLike + Default,
    D: DeconvolutedCentroidLike + Default,
    G: SpectrumGrouping<C, D, MultiLayerSpectrum<C, D>>,
> {
    group: G,
    pub targets: Vec<SelectedTarget>,
    _c: PhantomData<C>,
    _d: PhantomData<D>,
}

impl<T> SpectrumGroupTiming for T
where
    T: SpectrumGrouping<CPeak, DPeak, MultiLayerSpectrum<CPeak, DPeak>>,
{
    fn earliest_time(&self) -> Option<f64> {
        self.earliest_spectrum().map(|s| s.start_time())
    }

    fn latest_time(&self) -> Option<f64> {
        self.latest_spectrum().map(|s| s.start_time())
    }
}

impl<
        C: CentroidLike + Default,
        D: DeconvolutedCentroidLike + Default,
        G: SpectrumGrouping<C, D, MultiLayerSpectrum<C, D>>,
    > TargetTrackingSpectrumGroup<C, D, G>
{
    pub fn new(group: G, mut targets: Vec<SelectedTarget>) -> Self {
        targets.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());
        Self {
            group,
            targets,
            _c: PhantomData,
            _d: PhantomData,
        }
    }

    pub fn iter(
        &self,
    ) -> SpectrumGroupIter<'_, C, D, MultiLayerSpectrum<C, D>, TargetTrackingSpectrumGroup<C, D, G>>
    {
        SpectrumGroupIter::new(self)
    }

    pub fn selected_intervals(&self, mz_before: f64, mz_after: f64) -> Vec<(f64, f64)> {
        self.targets
            .iter()
            .map(|t| (t.mz - mz_before, t.mz + mz_after))
            .fold(Vec::new(), |mut acc, iv| {
                match acc.last_mut() {
                    Some(tail) => {
                        if SimpleInterval::from(*tail).overlaps(&SimpleInterval::from(iv)) {
                            tail.1 = iv.1;
                        } else {
                            acc.push(iv);
                        }
                    }
                    None => {
                        acc.push(iv);
                    }
                }

                acc
            })
    }
}

impl<
        C: CentroidLike + Default,
        D: DeconvolutedCentroidLike + Default,
        G: SpectrumGrouping<C, D, MultiLayerSpectrum<C, D>>,
    > IntoIterator for TargetTrackingSpectrumGroup<C, D, G>
{
    type Item = MultiLayerSpectrum<C, D>;

    type IntoIter = SpectrumGroupIntoIter<C, D, MultiLayerSpectrum<C, D>, Self>;

    fn into_iter(self) -> Self::IntoIter {
        SpectrumGroupIntoIter::new(self)
    }
}

impl<
        C: CentroidLike + Default,
        D: DeconvolutedCentroidLike + Default,
        G: SpectrumGrouping<C, D, MultiLayerSpectrum<C, D>>,
    > SpectrumGrouping<C, D, MultiLayerSpectrum<C, D>> for TargetTrackingSpectrumGroup<C, D, G>
{
    fn precursor(&self) -> Option<&MultiLayerSpectrum<C, D>> {
        <G as SpectrumGrouping<C, D, MultiLayerSpectrum<C, D>>>::precursor(&self.group)
    }

    fn precursor_mut(&mut self) -> Option<&mut MultiLayerSpectrum<C, D>> {
        <G as SpectrumGrouping<C, D, MultiLayerSpectrum<C, D>>>::precursor_mut(&mut self.group)
    }

    fn set_precursor(&mut self, prec: MultiLayerSpectrum<C, D>) {
        <G as SpectrumGrouping<C, D, MultiLayerSpectrum<C, D>>>::set_precursor(
            &mut self.group,
            prec,
        )
    }

    fn products(&self) -> &[MultiLayerSpectrum<C, D>] {
        <G as SpectrumGrouping<C, D, MultiLayerSpectrum<C, D>>>::products(&self.group)
    }

    fn products_mut(&mut self) -> &mut Vec<MultiLayerSpectrum<C, D>> {
        <G as SpectrumGrouping<C, D, MultiLayerSpectrum<C, D>>>::products_mut(&mut self.group)
    }

    fn total_spectra(&self) -> usize {
        <G as SpectrumGrouping<C, D, MultiLayerSpectrum<C, D>>>::total_spectra(&self.group)
    }

    fn earliest_spectrum(&self) -> Option<&MultiLayerSpectrum<C, D>> {
        <G as SpectrumGrouping<C, D, MultiLayerSpectrum<C, D>>>::earliest_spectrum(&self.group)
    }

    fn latest_spectrum(&self) -> Option<&MultiLayerSpectrum<C, D>> {
        <G as SpectrumGrouping<C, D, MultiLayerSpectrum<C, D>>>::latest_spectrum(&self.group)
    }

    fn lowest_ms_level(&self) -> Option<u8> {
        <G as SpectrumGrouping<C, D, MultiLayerSpectrum<C, D>>>::lowest_ms_level(&self.group)
    }

    fn highest_ms_level(&self) -> Option<u8> {
        <G as SpectrumGrouping<C, D, MultiLayerSpectrum<C, D>>>::highest_ms_level(&self.group)
    }

    fn into_parts(
        self,
    ) -> (
        Option<MultiLayerSpectrum<C, D>>,
        Vec<MultiLayerSpectrum<C, D>>,
    ) {
        <G as SpectrumGrouping<C, D, MultiLayerSpectrum<C, D>>>::into_parts(self.group)
    }
}

type SpectrumGroupWithStartTime<C, D> = (SpectrumGroup<C, D, MultiLayerSpectrum<C, D>>, f64);
type SpectrumGroupWithTargets<C, D> = (
    SpectrumGroup<C, D, MultiLayerSpectrum<C, D>>,
    Vec<SelectedTarget>,
);

pub struct MSnTargetTrackingIterator<
    C: CentroidLike + Default,
    D: DeconvolutedCentroidLike + Default,
    R: Iterator<Item = SpectrumGroup<C, D, MultiLayerSpectrum<C, D>>>,
> {
    source: R,
    time_width: f64,
    error_tolerance: Tolerance,
    buffer: VecDeque<SpectrumGroupWithStartTime<C, D>>,
    pushback_buffer: Option<SpectrumGroupWithStartTime<C, D>>,
    targets: VecDeque<SelectionTargetSpecification>,
}

impl<
        C: CentroidLike + Default,
        D: DeconvolutedCentroidLike + Default,
        R: Iterator<Item = SpectrumGroup<C, D, MultiLayerSpectrum<C, D>>>,
    > Iterator for MSnTargetTrackingIterator<C, D, R>
{
    type Item = TargetTrackingSpectrumGroup<C, D, SpectrumGroup<C, D, MultiLayerSpectrum<C, D>>>;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.step();
        self.feed_next();
        value.map(|(g, ts)| TargetTrackingSpectrumGroup::new(g, ts))
    }
}

impl<
        C: CentroidLike + Default,
        D: DeconvolutedCentroidLike + Default,
        R: Iterator<Item = SpectrumGroup<C, D, MultiLayerSpectrum<C, D>>>,
    > MSnTargetTrackingIterator<C, D, R>
{
    pub fn new(source: R, time_width: f64, error_tolerance: Tolerance) -> Self {
        let mut inst = Self {
            source,
            time_width,
            error_tolerance,
            buffer: Default::default(),
            pushback_buffer: Default::default(),
            targets: Default::default(),
        };
        inst.initial_feed();
        inst
    }

    fn observe(&mut self, group: &SpectrumGroup<C, D, MultiLayerSpectrum<C, D>>) {
        let error_tolerance = self.error_tolerance;
        let time_width_half = self.time_width / 2.0;
        let pad = 0.5f64.min(time_width_half);
        let new_targets: usize = group
            .products
            .iter()
            .map(|s| {
                let prec = s.precursor().unwrap();
                let mz = prec.mz();
                let t = s.start_time();
                let hits: usize = self
                    .targets
                    .iter_mut()
                    .map(|p| {
                        if error_tolerance.test(p.mz, mz)
                            && (t > p.time_range.end)
                            && (t - p.time_range.end) < time_width_half
                        {
                            p.time_range.end = t;
                            1
                        } else {
                            0
                        }
                    })
                    .sum();
                if hits == 0 {
                    let p = SelectionTargetSpecification::new(
                        mz,
                        prec.charge(),
                        (t - pad..t + pad).into(),
                    );
                    self.targets.push_back(p);
                    1
                } else {
                    0
                }
            })
            .sum();
        if new_targets > 0 {
            tracing::trace!("Added {new_targets} new targets");
        }
    }

    fn initial_feed(&mut self) {
        let group = self.source.next();
        tracing::trace!("First spectrum group was found? {}", group.is_some());
        let start_time = group
            .as_ref()
            .and_then(|s| {
                s.earliest_spectrum().inspect(|s| {
                    tracing::trace!("Earliest spectrum {} occurs at {}", s.id(), s.start_time());
                }).map(|s| s.start_time())
            })
            .unwrap();
        let end_time = start_time + self.time_width;
        tracing::trace!("Initial time window {start_time} to {end_time}");
        if let Some(g) = group {
            self.observe(&g);
            self.buffer.push_back((g, start_time));
        }
        while let Some(group) = self.source.next() {
            let t = group.earliest_spectrum().map(|s| s.start_time()).unwrap();
            if t < end_time {
                self.observe(&group);
                self.buffer.push_back((group, t));
            } else {
                self.pushback_buffer = Some((group, t));
                break;
            }
        }
        tracing::trace!(
            "{} targets extracted from buffer size {}",
            self.targets.len(),
            self.buffer.len()
        );
    }

    fn get_current_window_end(&self) -> f64 {
        if self.buffer.is_empty() {
            return f64::NEG_INFINITY;
        }
        let start = self.buffer.front().map(|(_, t)| *t).unwrap();
        let end = self.buffer.back().map(|(_, t)| *t).unwrap();
        let mid = start + (end - start) / 2.0;
        mid + self.time_width
    }

    fn feed_next(&mut self) {
        let threshold = self.get_current_window_end();
        let (use_pushback, pushback_populated) = if let Some((_, t)) = self.pushback_buffer.as_ref()
        {
            (*t < threshold, true)
        } else {
            (false, false)
        };
        if use_pushback {
            let (group, t) = self.pushback_buffer.take().unwrap();
            self.observe(&group);
            self.buffer.push_back((group, t));
        }
        if !pushback_populated {
            if let Some(group) = self.source.next() {
                let t = group.earliest_spectrum().map(|s| s.start_time()).unwrap();
                if t < threshold {
                    self.observe(&group);
                    self.buffer.push_back((group, t));
                } else {
                    self.pushback_buffer = Some((group, t));
                }
            }
        }
    }

    fn step(&mut self) -> Option<SpectrumGroupWithTargets<C, D>> {
        if let Some((group, t)) = self.buffer.pop_front() {
            let targets: Vec<_> = self
                .targets
                .iter()
                .filter(|p| p.spans(t))
                .map(|p| SelectedTarget::new(p.mz, p.charge))
                .collect();
            self.targets = self
                .targets
                .drain(..)
                .filter(|target| {
                    // Keep targets which have not ended by this time point
                    let cond = target.time_range.end >= t;
                    if !cond {
                        tracing::trace!("Dropping {target:?} at {t}")
                    }
                    cond
                })
                .collect();
            Some((group, targets))
        } else {
            None
        }
    }
}

pub trait MSnTargetTracking<C: CentroidLike + Default, D: DeconvolutedCentroidLike + Default>:
    Iterator<Item = SpectrumGroup<C, D, MultiLayerSpectrum<C, D>>> + Sized
{
    fn track_precursors(
        self,
        time_width: f64,
        error_tolerance: Tolerance,
    ) -> MSnTargetTrackingIterator<C, D, Self> {
        MSnTargetTrackingIterator::new(self, time_width, error_tolerance)
    }
}

impl<C: CentroidLike + Default, D: DeconvolutedCentroidLike + Default, T> MSnTargetTracking<C, D>
    for T
where
    T: Iterator<Item = SpectrumGroup<C, D, MultiLayerSpectrum<C, D>>> + Sized,
{
}

#[derive(Debug)]
pub struct TargetTrackingFrameGroup<
    C: FeatureLike<MZ, IonMobility> + Default,
    D: FeatureLike<Mass, IonMobility> + KnownCharge + Default,
    G: IonMobilityFrameGrouping<C, D, MultiLayerIonMobilityFrame<C, D>>,
> {
    pub(crate) group: G,
    pub targets: Vec<SelectedTarget>,
    _c: PhantomData<C>,
    _d: PhantomData<D>,
}

impl<
        C: FeatureLike<MZ, IonMobility> + Default,
        D: FeatureLike<Mass, IonMobility> + KnownCharge + Default,
        G: IonMobilityFrameGrouping<C, D, MultiLayerIonMobilityFrame<C, D>>,
    > Default for TargetTrackingFrameGroup<C, D, G>
{
    fn default() -> Self {
        Self {
            group: Default::default(),
            targets: Default::default(),
            _c: Default::default(),
            _d: Default::default(),
        }
    }
}

impl<
        C: FeatureLike<MZ, IonMobility> + Default,
        D: FeatureLike<Mass, IonMobility> + KnownCharge + Default,
        G: IonMobilityFrameGrouping<C, D, MultiLayerIonMobilityFrame<C, D>>,
    > IntoIterator for TargetTrackingFrameGroup<C, D, G>
{
    type Item = MultiLayerIonMobilityFrame<C, D>;

    type IntoIter = IonMobilityFrameGroupIntoIter<C, D, Self::Item, G>;

    fn into_iter(self) -> Self::IntoIter {
        IonMobilityFrameGroupIntoIter::new(self.group)
    }
}

impl<
        C: FeatureLike<MZ, IonMobility> + Default,
        D: FeatureLike<Mass, IonMobility> + KnownCharge + Default,
        G: IonMobilityFrameGrouping<C, D, MultiLayerIonMobilityFrame<C, D>>,
    > TargetTrackingFrameGroup<C, D, G>
{
    pub fn new(group: G, targets: Vec<SelectedTarget>) -> Self {
        Self {
            group,
            targets,
            _c: PhantomData,
            _d: PhantomData,
        }
    }

    #[allow(unused)]
    pub fn iter(&self) -> IonMobilityFrameGroupIter<'_, C, D, MultiLayerIonMobilityFrame<C, D>, G> {
        IonMobilityFrameGroupIter::new(&self.group)
    }

    pub fn selected_intervals(&self, mz_before: f64, mz_after: f64) -> Vec<(f64, f64)> {
        self.targets
            .iter()
            .map(|t| (t.mz - mz_before, t.mz + mz_after))
            .fold(Vec::new(), |mut acc, iv| {
                match acc.last_mut() {
                    Some(tail) => {
                        if SimpleInterval::from(*tail).overlaps(&SimpleInterval::from(iv)) {
                            tail.1 = iv.1;
                        } else {
                            acc.push(iv);
                        }
                    }
                    None => {
                        acc.push(iv);
                    }
                }

                acc
            })
    }
}

impl<
        C: FeatureLike<MZ, IonMobility> + Default,
        D: FeatureLike<Mass, IonMobility> + KnownCharge + Default,
        G: IonMobilityFrameGrouping<C, D, MultiLayerIonMobilityFrame<C, D>>,
    > IonMobilityFrameGrouping<C, D, MultiLayerIonMobilityFrame<C, D>>
    for TargetTrackingFrameGroup<C, D, G>
{
    fn precursor(&self) -> Option<&MultiLayerIonMobilityFrame<C, D>> {
        <G as IonMobilityFrameGrouping<C, D, MultiLayerIonMobilityFrame<C, D>>>::precursor(
            &self.group,
        )
    }

    fn precursor_mut(&mut self) -> Option<&mut MultiLayerIonMobilityFrame<C, D>> {
        <G as IonMobilityFrameGrouping<C, D, MultiLayerIonMobilityFrame<C, D>>>::precursor_mut(
            &mut self.group,
        )
    }

    fn set_precursor(&mut self, prec: MultiLayerIonMobilityFrame<C, D>) {
        <G as IonMobilityFrameGrouping<C, D, MultiLayerIonMobilityFrame<C, D>>>::set_precursor(
            &mut self.group,
            prec,
        )
    }

    fn products(&self) -> &[MultiLayerIonMobilityFrame<C, D>] {
        <G as IonMobilityFrameGrouping<C, D, MultiLayerIonMobilityFrame<C, D>>>::products(
            &self.group,
        )
    }

    fn products_mut(&mut self) -> &mut Vec<MultiLayerIonMobilityFrame<C, D>> {
        <G as IonMobilityFrameGrouping<C, D, MultiLayerIonMobilityFrame<C, D>>>::products_mut(
            &mut self.group,
        )
    }

    fn total_frames(&self) -> usize {
        <G as IonMobilityFrameGrouping<C, D, MultiLayerIonMobilityFrame<C, D>>>::total_frames(
            &self.group,
        )
    }

    fn earliest_frame(&self) -> Option<&MultiLayerIonMobilityFrame<C, D>> {
        <G as IonMobilityFrameGrouping<C, D, MultiLayerIonMobilityFrame<C, D>>>::earliest_frame(
            &self.group,
        )
    }

    fn latest_frame(&self) -> Option<&MultiLayerIonMobilityFrame<C, D>> {
        <G as IonMobilityFrameGrouping<C, D, MultiLayerIonMobilityFrame<C, D>>>::latest_frame(
            &self.group,
        )
    }

    fn lowest_ms_level(&self) -> Option<u8> {
        <G as IonMobilityFrameGrouping<C, D, MultiLayerIonMobilityFrame<C, D>>>::lowest_ms_level(
            &self.group,
        )
    }

    fn highest_ms_level(&self) -> Option<u8> {
        <G as IonMobilityFrameGrouping<C, D, MultiLayerIonMobilityFrame<C, D>>>::highest_ms_level(
            &self.group,
        )
    }

    fn into_parts(
        self,
    ) -> (
        Option<MultiLayerIonMobilityFrame<C, D>>,
        Vec<MultiLayerIonMobilityFrame<C, D>>,
    ) {
        <G as IonMobilityFrameGrouping<C, D, MultiLayerIonMobilityFrame<C, D>>>::into_parts(
            self.group,
        )
    }
}

type IonMobilityFrameGroupWithStartTime<C, D> = (
    IonMobilityFrameGroup<C, D, MultiLayerIonMobilityFrame<C, D>>,
    f64,
);
type IonMobilityFrameGroupWithTargets<C, D> = (
    IonMobilityFrameGroup<C, D, MultiLayerIonMobilityFrame<C, D>>,
    Vec<SelectedTarget>,
);

pub struct MSnTargetTrackingFrameIterator<
    C: FeatureLike<MZ, IonMobility> + Default,
    D: FeatureLike<Mass, IonMobility> + KnownCharge + Default,
    R: Iterator<Item = IonMobilityFrameGroup<C, D, MultiLayerIonMobilityFrame<C, D>>>,
> {
    source: R,
    time_width: f64,
    error_tolerance: Tolerance,
    buffer: VecDeque<IonMobilityFrameGroupWithStartTime<C, D>>,
    pushback_buffer: Option<IonMobilityFrameGroupWithStartTime<C, D>>,
    targets: VecDeque<SelectionTargetSpecification>,
}

impl<
        C: FeatureLike<MZ, IonMobility> + Default,
        D: FeatureLike<Mass, IonMobility> + KnownCharge + Default,
        R: Iterator<Item = IonMobilityFrameGroup<C, D, MultiLayerIonMobilityFrame<C, D>>>,
    > MSnTargetTrackingFrameIterator<C, D, R>
{
    pub fn new(source: R, time_width: f64, error_tolerance: Tolerance) -> Self {
        let mut inst = Self {
            source,
            time_width,
            error_tolerance,
            buffer: Default::default(),
            pushback_buffer: Default::default(),
            targets: Default::default(),
        };
        inst.initial_feed();
        inst
    }

    fn observe(&mut self, group: &IonMobilityFrameGroup<C, D, MultiLayerIonMobilityFrame<C, D>>) {
        let error_tolerance = self.error_tolerance;
        let time_width_half = self.time_width / 2.0;
        let pad = 0.5f64.min(time_width_half);
        let new_targets: usize = group
            .products
            .iter()
            .map(|s| {
                if let Some(prec) = s.precursor() {
                    let mz = prec.mz();
                    let t = s.start_time();
                    let hits: usize = self
                        .targets
                        .iter_mut()
                        .map(|p| {
                            if error_tolerance.test(p.mz, mz)
                                && (t > p.time_range.end)
                                && (t - p.time_range.end) < time_width_half
                            {
                                p.time_range.end = t;
                                1
                            } else {
                                0
                            }
                        })
                        .sum();
                    if hits == 0 {
                        let p = SelectionTargetSpecification::new(
                            mz,
                            prec.charge(),
                            (t - pad..t + pad).into(),
                        );
                        self.targets.push_back(p);
                        1
                    } else {
                        0
                    }
                }
                else {
                    0
                }
            })
            .sum();
        if new_targets > 0 {
            tracing::trace!("Added {new_targets} new targets");
        }
    }

    fn initial_feed(&mut self) {
        let group = self.source.next();
        let start_time = group
            .as_ref()
            .and_then(|s| s.earliest_frame().map(|s| s.start_time()))
            .unwrap();
        let end_time = start_time + self.time_width;
        tracing::trace!("Initial time window {start_time} to {end_time}");
        if let Some(g) = group {
            self.observe(&g);
            self.buffer.push_back((g, start_time));
        }
        while let Some(group) = self.source.next() {
            let t = group.earliest_frame().map(|s| s.start_time()).unwrap();
            if t < end_time {
                self.observe(&group);
                self.buffer.push_back((group, t));
            } else {
                self.pushback_buffer = Some((group, t));
                break;
            }
        }
        tracing::trace!(
            "{} targets extracted from buffer size {}",
            self.targets.len(),
            self.buffer.len()
        );
    }

    fn get_current_window_end(&self) -> f64 {
        if self.buffer.is_empty() {
            return f64::NEG_INFINITY;
        }
        let start = self.buffer.front().map(|(_, t)| *t).unwrap();
        let end = self.buffer.back().map(|(_, t)| *t).unwrap();
        let mid = start + (end - start) / 2.0;
        mid + self.time_width
    }

    fn feed_next(&mut self) {
        let threshold = self.get_current_window_end();
        let (use_pushback, pushback_populated) = if let Some((_, t)) = self.pushback_buffer.as_ref()
        {
            (*t < threshold, true)
        } else {
            (false, false)
        };
        if use_pushback {
            let (group, t) = self.pushback_buffer.take().unwrap();
            self.observe(&group);
            self.buffer.push_back((group, t));
        }
        if !pushback_populated {
            if let Some(group) = self.source.next() {
                let t = group.earliest_frame().map(|s| s.start_time()).unwrap();
                if t < threshold {
                    self.observe(&group);
                    self.buffer.push_back((group, t));
                } else {
                    self.pushback_buffer = Some((group, t));
                }
            }
        }
    }

    fn step(&mut self) -> Option<IonMobilityFrameGroupWithTargets<C, D>> {
        if let Some((group, t)) = self.buffer.pop_front() {
            let targets: Vec<_> = self
                .targets
                .iter()
                .filter(|p| p.spans(t))
                .map(|p| SelectedTarget::new(p.mz, p.charge))
                .collect();
            self.targets = self
                .targets
                .drain(..)
                .filter(|target| {
                    // Keep targets which have not ended by this time point
                    let cond = target.time_range.end >= t;
                    if !cond {
                        tracing::trace!("Dropping {target:?} at {t}")
                    }
                    cond
                })
                .collect();
            Some((group, targets))
        } else {
            None
        }
    }
}

impl<
        C: FeatureLike<MZ, IonMobility> + Default,
        D: FeatureLike<Mass, IonMobility> + KnownCharge + Default,
        R: Iterator<Item = IonMobilityFrameGroup<C, D, MultiLayerIonMobilityFrame<C, D>>>,
    > Iterator for MSnTargetTrackingFrameIterator<C, D, R>
{
    type Item = TargetTrackingFrameGroup<
        C,
        D,
        IonMobilityFrameGroup<C, D, MultiLayerIonMobilityFrame<C, D>>,
    >;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.step();
        self.feed_next();
        value.map(|(g, ts)| TargetTrackingFrameGroup::new(g, ts))
    }
}

pub trait FrameMSnTargetTracking<
    C: FeatureLike<MZ, IonMobility> + Default,
    D: FeatureLike<Mass, IonMobility> + KnownCharge + Default,
>: Iterator<Item = IonMobilityFrameGroup<C, D, MultiLayerIonMobilityFrame<C, D>>> + Sized
{
    fn track_precursors(
        self,
        time_width: f64,
        error_tolerance: Tolerance,
    ) -> MSnTargetTrackingFrameIterator<C, D, Self> {
        MSnTargetTrackingFrameIterator::new(self, time_width, error_tolerance)
    }
}

impl<
        C: FeatureLike<MZ, IonMobility> + Default,
        D: FeatureLike<Mass, IonMobility> + KnownCharge + Default,
        T,
    > FrameMSnTargetTracking<C, D> for T
where
    T: Iterator<Item = IonMobilityFrameGroup<C, D, MultiLayerIonMobilityFrame<C, D>>> + Sized,
{
}
