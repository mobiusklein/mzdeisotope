use mzdeisotope::{scorer::ScoreType, solution::DeconvolvedSolutionPeak};
use std::{
    boxed::Box,
    ops::{Bound, RangeBounds},
};

use mzpeaks::{
    feature::{ChargedFeature, FeatureView, TimeArray},
    feature_map::FeatureMap,
    peak::MZPoint,
    prelude::*,
    Mass, MZ,
};

use mzsignal::feature_mapping::{ChargeAwareFeatureMerger, FeatureGraphBuilder};

#[derive(Default, Debug, Clone)]
pub struct MZPointSeries {
    mz: Vec<f64>,
    intensity: Vec<f32>,
}

impl<'a> MZPointSeries {
    pub fn new(mz: Vec<f64>, intensity: Vec<f32>) -> Self {
        Self { mz, intensity }
    }

    pub fn push<T: CoordinateLike<MZ> + IntensityMeasurement>(&mut self, pt: T) {
        self.push_raw(pt.mz(), pt.intensity());
    }

    pub fn push_raw(&mut self, mz: f64, intensity: f32) {
        self.mz.push(mz);
        self.intensity.push(intensity);
    }

    pub fn split_at(&self, i: usize) -> (Self, Self) {
        let mz_a = self.mz[..i].to_vec();
        let mz_b = self.mz[i..].to_vec();

        let inten_a = self.intensity[..i].to_vec();
        let inten_b = self.intensity[i..].to_vec();

        (Self::new(mz_a, inten_a), Self::new(mz_b, inten_b))
    }

    pub fn slice<I: RangeBounds<usize> + Clone>(&self, bounds: I) -> Self {
        let start = match bounds.start_bound() {
            Bound::Included(i) | Bound::Excluded(i) => *i,
            Bound::Unbounded => 0,
        };

        let end = match bounds.end_bound() {
            Bound::Included(i) => *i + 1,
            Bound::Excluded(i) => *i,
            Bound::Unbounded => self.mz.len(),
        };

        Self::new(
            self.mz[start..end].to_vec(),
            self.intensity[start..end].to_vec(),
        )
    }

    pub fn len(&self) -> usize {
        self.mz.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mz.is_empty()
    }

    pub fn at(&self, index: usize) -> Option<MZPoint> {
        if index < self.len() {
            Some(MZPoint::new(self.mz[index], self.intensity[index]))
        } else {
            None
        }
    }

    pub fn as_feature_view<Y>(&'a self, time: &'a [f64]) -> FeatureView<'a, MZ, Y> {
        let start = 0;
        let end = time.len().min(self.len());

        FeatureView::new(&self.mz[start..end], &time, &self.intensity[start..end])
    }
}

#[derive(Debug, Default, Clone)]
pub struct DeconvolvedSolutionFeature<Y: Clone> {
    inner: ChargedFeature<Mass, Y>,
    pub score: ScoreType,
    pub scores: Vec<ScoreType>,
    envelope: Box<[MZPointSeries]>,
}

impl<Y: Clone> DeconvolvedSolutionFeature<Y> {
    pub fn new(
        inner: ChargedFeature<Mass, Y>,
        score: ScoreType,
        scores: Vec<ScoreType>,
        envelope: Box<[MZPointSeries]>,
    ) -> Self {
        Self {
            inner,
            score,
            scores,
            envelope,
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn iter(&self) -> mzpeaks::feature::Iter<'_, Mass, Y> {
        self.inner.iter()
    }

    pub fn iter_mut(&mut self) -> mzpeaks::feature::IterMut<'_, Mass, Y> {
        self.inner.iter_mut()
    }

    pub fn iter_peaks(&self) -> PeakIter<'_, Y> {
        PeakIter::new(&self)
    }

    pub fn iter_envelope(&self) -> EnvelopeIter<'_, Y> {
        EnvelopeIter::new(self)
    }

    pub fn push<T: CoordinateLike<Mass> + IntensityMeasurement>(&mut self, pt: &T, time: f64) {
        self.inner.push(pt, time);
        self.scores.push(0.0);
    }

    pub fn push_peak(&mut self, peak: &DeconvolvedSolutionPeak, time: f64) {
        self.inner.push_raw(peak.neutral_mass, time, peak.intensity);
        self.scores.push(peak.score);
        self.envelope
            .iter_mut()
            .zip(peak.envelope.iter())
            .for_each(|(ev, pt)| ev.push(pt));
    }

    pub fn envelope(&self) -> Vec<FeatureView<MZ, Y>> {
        let times = self.inner.time_view();
        self.envelope
            .iter()
            .map(|s| s.as_feature_view(times))
            .collect()
    }
}

impl<Y0: Clone> AsRef<ChargedFeature<Mass, Y0>> for DeconvolvedSolutionFeature<Y0> {
    fn as_ref(&self) -> &ChargedFeature<Mass, Y0> {
        &self.inner
    }
}

impl<Y0: Clone> AsMut<ChargedFeature<Mass, Y0>> for DeconvolvedSolutionFeature<Y0> {
    fn as_mut(&mut self) -> &mut ChargedFeature<Mass, Y0> {
        &mut self.inner
    }
}

impl<Y0: Clone> FeatureLikeMut<Mass, Y0> for DeconvolvedSolutionFeature<Y0> {
    fn iter_mut(&mut self) -> impl Iterator<Item = (&mut f64, &mut f64, &mut f32)> {
        <ChargedFeature<Mass, Y0> as FeatureLikeMut<Mass, Y0>>::iter_mut(&mut self.inner)
    }

    fn push<T: CoordinateLike<Mass> + IntensityMeasurement>(&mut self, pt: &T, time: f64) {
        <ChargedFeature<Mass, Y0> as FeatureLikeMut<Mass, Y0>>::push(&mut self.inner, pt, time)
    }

    fn push_raw(&mut self, x: f64, y: f64, z: f32) {
        <ChargedFeature<Mass, Y0> as FeatureLikeMut<Mass, Y0>>::push_raw(&mut self.inner, x, y, z)
    }
}

impl<Y0: Clone> TimeInterval<Y0> for DeconvolvedSolutionFeature<Y0> {
    fn apex_time(&self) -> Option<f64> {
        <ChargedFeature<Mass, Y0> as TimeInterval<Y0>>::apex_time(&self.inner)
    }

    fn area(&self) -> f32 {
        <ChargedFeature<Mass, Y0> as TimeInterval<Y0>>::area(&self.inner)
    }

    fn end_time(&self) -> Option<f64> {
        <ChargedFeature<Mass, Y0> as TimeInterval<Y0>>::end_time(&self.inner)
    }

    fn start_time(&self) -> Option<f64> {
        <ChargedFeature<Mass, Y0> as TimeInterval<Y0>>::start_time(&self.inner)
    }

    fn iter_time(&self) -> impl Iterator<Item = f64> {
        <ChargedFeature<Mass, Y0> as TimeInterval<Y0>>::iter_time(&self.inner)
    }

    fn find_time(&self, time: f64) -> (Option<usize>, f64) {
        <ChargedFeature<Mass, Y0> as TimeInterval<Y0>>::find_time(&self.inner, time)
    }
}

impl<Y0: Clone> TimeArray<Y0> for DeconvolvedSolutionFeature<Y0> {
    fn time_view(&self) -> &[f64] {
        self.inner.time_view()
    }
}

impl<Y0: Clone> FeatureLike<Mass, Y0> for DeconvolvedSolutionFeature<Y0> {
    fn len(&self) -> usize {
        <ChargedFeature<Mass, Y0> as FeatureLike<Mass, Y0>>::len(&self.inner)
    }

    fn iter(&self) -> impl Iterator<Item = (&f64, &f64, &f32)> {
        <ChargedFeature<Mass, Y0> as FeatureLike<Mass, Y0>>::iter(&self.inner)
    }
}

impl<Y: Clone> PartialOrd for DeconvolvedSolutionFeature<Y> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.inner.partial_cmp(&other.inner) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        self.score.partial_cmp(&other.score)
    }
}

impl<Y: Clone> PartialEq for DeconvolvedSolutionFeature<Y> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner && self.score == other.score
    }
}

impl<Y0: Clone> CoordinateLike<Mass> for DeconvolvedSolutionFeature<Y0> {
    fn coordinate(&self) -> f64 {
        <ChargedFeature<Mass, Y0> as CoordinateLike<Mass>>::coordinate(&self.inner)
    }
}

impl<Y0: Clone> KnownCharge for DeconvolvedSolutionFeature<Y0> {
    fn charge(&self) -> i32 {
        <ChargedFeature<Mass, Y0> as KnownCharge>::charge(&self.inner)
    }
}

impl<Y0: Clone> IntensityMeasurement for DeconvolvedSolutionFeature<Y0> {
    fn intensity(&self) -> f32 {
        <ChargedFeature<Mass, Y0> as IntensityMeasurement>::intensity(&self.inner)
    }
}

impl<Y: Clone> SplittableFeatureLike<'_, Mass, Y> for DeconvolvedSolutionFeature<Y> {
    type ViewType = DeconvolvedSolutionFeature<Y>;

    fn split_at_time(&self, point: f64) -> (Self::ViewType, Self::ViewType) {
        if let (Some(idx), _) = self.find_time(point) {
            let (before, after) = self.inner.split_at_time(point);
            let mut envelope_before = Vec::new();
            let mut envelope_after = Vec::new();
            for (env_before_i, env_after_i) in self.envelope.iter().map(|e| {
                let (a, b) = e.split_at(idx);
                (a.to_owned(), b.to_owned())
            }) {
                envelope_before.push(env_before_i);
                envelope_after.push(env_after_i);
            }
            (
                Self::new(
                    before.to_owned(),
                    self.score,
                    self.scores[..idx].to_vec(),
                    envelope_before.into_boxed_slice(),
                ),
                Self::new(
                    after.to_owned(),
                    self.score,
                    self.scores[idx..].to_vec(),
                    envelope_after.into_boxed_slice(),
                ),
            )
        } else {
            let mut envelope_before = Vec::new();
            let mut envelope_after = Vec::new();
            for (env_before_i, env_after_i) in self
                .envelope
                .iter()
                .map(|_| (MZPointSeries::default(), MZPointSeries::default()))
            {
                envelope_before.push(env_before_i);
                envelope_after.push(env_after_i);
            }
            return (
                Self::new(
                    ChargedFeature::empty(self.charge()),
                    self.score,
                    Vec::new(),
                    envelope_before.into_boxed_slice(),
                ),
                Self::new(
                    ChargedFeature::empty(self.charge()),
                    self.score,
                    Vec::new(),
                    envelope_after.into_boxed_slice(),
                ),
            );
        }
    }

    fn split_at(&self, point: usize) -> (Self::ViewType, Self::ViewType) {
        let (before, after) = self.inner.split_at(point);
        let mut envelope_before = Vec::new();
        let mut envelope_after = Vec::new();
        for (env_before_i, env_after_i) in self.envelope.iter().map(|e| {
            let (a, b) = e.split_at(point);
            (a.to_owned(), b.to_owned())
        }) {
            envelope_before.push(env_before_i);
            envelope_after.push(env_after_i);
        }
        (
            Self::new(
                before.to_owned(),
                self.score,
                self.scores[..point].to_vec(),
                envelope_before.into_boxed_slice(),
            ),
            Self::new(
                after.to_owned(),
                self.score,
                self.scores[point..].to_vec(),
                envelope_after.into_boxed_slice(),
            ),
        )
    }

    fn slice<I: std::ops::RangeBounds<usize> + Clone>(&self, bounds: I) -> Self::ViewType {
        let inner = self.inner.slice(bounds.clone()).to_owned();
        let envelope: Vec<_> = self
            .envelope
            .iter()
            .map(|e| e.slice(bounds.clone()).to_owned())
            .collect();

        let start = match bounds.start_bound() {
            Bound::Included(i) => *i,
            Bound::Excluded(i) => *i,
            Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            Bound::Included(i) => *i + 1,
            Bound::Excluded(i) => *i,
            Bound::Unbounded => self.scores.len(),
        };

        let scores = self.scores[start..end].to_vec();

        Self::new(inner, self.score, scores, envelope.into_boxed_slice())
    }
}

pub struct PeakIter<'a, Y: Clone> {
    feature: &'a DeconvolvedSolutionFeature<Y>,
    i: usize,
}

impl<'a, Y: Clone> PeakIter<'a, Y> {
    pub fn new(feature: &'a DeconvolvedSolutionFeature<Y>) -> Self {
        Self { feature, i: 0 }
    }
}

impl<'a, Y: Clone> Iterator for PeakIter<'a, Y> {
    type Item = (DeconvolvedSolutionPeak, f64);

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        if i < self.feature.len() {
            let (mass, time, inten) = self.feature.at(i).unwrap();
            let score = self.feature.scores[i];
            let env: Vec<_> = self
                .feature
                .envelope
                .iter()
                .map(|e| e.at(i).unwrap())
                .collect();
            let peak = DeconvolvedSolutionPeak::new(
                mass,
                inten,
                self.feature.charge(),
                0,
                score,
                Box::new(env),
            );
            self.i += 1;
            Some((peak, time))
        } else {
            None
        }
    }
}

pub struct EnvelopeIter<'a, Y: Clone> {
    feature: &'a DeconvolvedSolutionFeature<Y>,
    i: usize,
}

impl<'a, Y: Clone> EnvelopeIter<'a, Y> {
    pub fn new(feature: &'a DeconvolvedSolutionFeature<Y>) -> Self {
        Self { feature, i: 0 }
    }
}

impl<'a, Y: Clone> Iterator for EnvelopeIter<'a, Y> {
    type Item = (f64, Box<[(f64, f32)]>);

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        if i < self.feature.len() {
            let (_, time, _) = self.feature.at(i).unwrap();
            let env = self
                .feature
                .envelope
                .iter()
                .map(|e| {
                    let pt = e.at(i).unwrap();
                    (pt.mz, pt.intensity)
                })
                .collect();
            self.i += 1;
            Some((time, env))
        } else {
            None
        }
    }
}

#[derive(Default)]
pub struct FeatureMerger<Y: Clone + Default> {
    inner: ChargeAwareFeatureMerger<Mass, Y, DeconvolvedSolutionFeature<Y>>,
}

impl<Y: Clone + Default> FeatureMerger<Y> {}

impl<Y: Clone + Default> FeatureGraphBuilder<Mass, Y, DeconvolvedSolutionFeature<Y>>
    for FeatureMerger<Y>
{
    fn build_graph(
        &self,
        features: &mzpeaks::feature_map::FeatureMap<Mass, Y, DeconvolvedSolutionFeature<Y>>,
        mass_error_tolerance: Tolerance,
        maximum_gap_size: f64,
    ) -> Vec<mzsignal::feature_mapping::FeatureNode> {
        self.inner
            .build_graph(features, mass_error_tolerance, maximum_gap_size)
    }

    fn merge_components(
        &self,
        features: &FeatureMap<Mass, Y, DeconvolvedSolutionFeature<Y>>,
        connected_components: Vec<Vec<usize>>,
    ) -> FeatureMap<Mass, Y, DeconvolvedSolutionFeature<Y>> {
        let mut merged_nodes = Vec::new();
        for component_indices in connected_components {
            if component_indices.is_empty() {
                continue;
            }
            let mut features_of: Vec<_> = component_indices
                .into_iter()
                .map(|i| features.get_item(i))
                .collect();
            features_of.sort_by(|a, b| a.start_time().unwrap().total_cmp(&b.start_time().unwrap()));
            let mut acc = (*features_of[0]).clone();
            for f in &features_of[1..] {
                debug_assert_eq!(acc.charge(), f.charge());
                for (peak, time) in f.iter_peaks() {
                    acc.push_peak(&peak, time);
                }
            }
            merged_nodes.push(acc);
        }

        FeatureMap::new(merged_nodes)
    }
}
