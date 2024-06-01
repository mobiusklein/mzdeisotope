use mzdeisotope::scorer::ScoreType;
use std::boxed::Box;

use mzpeaks::{
    feature::{ChargedFeature, Feature}, prelude::*, Mass, MZ
};

#[derive(Debug, Default, Clone)]
pub struct DeconvolvedSolutionFeature<Y: Clone> {
    inner: ChargedFeature<Mass, Y>,
    pub score: ScoreType,
    pub envelope: Box<[Feature<MZ, Y>]>,
}

impl<Y: Clone> DeconvolvedSolutionFeature<Y> {
    pub fn new(
        inner: ChargedFeature<Mass, Y>,
        score: ScoreType,
        envelope: Box<[Feature<MZ, Y>]>,
    ) -> Self {
        Self {
            inner,
            score,
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

    pub fn iter_peaks(&self) -> mzpeaks::feature::DeconvolutedPeakIter<'_, Y> {
        self.inner.iter_peaks()
    }

    pub fn push<T: CoordinateLike<Mass> + IntensityMeasurement>(&mut self, pt: &T, time: f64) {
        self.inner.push(pt, time)
    }
}

impl<Y0: Clone> AsRef<mzpeaks::feature::ChargedFeature<Mass, Y0>>
    for DeconvolvedSolutionFeature<Y0>
{
    fn as_ref(&self) -> &mzpeaks::feature::ChargedFeature<Mass, Y0> {
        &self.inner
    }
}

impl<Y0: Clone> AsMut<mzpeaks::feature::ChargedFeature<Mass, Y0>>
    for DeconvolvedSolutionFeature<Y0>
{
    fn as_mut(&mut self) -> &mut mzpeaks::feature::ChargedFeature<Mass, Y0> {
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

    fn split_at(&self, point: f64) -> (Self::ViewType, Self::ViewType) {
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
            Self::new(before.to_owned(), self.score, envelope_before.into_boxed_slice()),
            Self::new(after.to_owned(), self.score, envelope_after.into_boxed_slice())
        )
    }

    fn slice(&self, bounds: impl std::ops::RangeBounds<usize>) -> Self::ViewType {
        let n = self.len();
        let inner = self.inner.slice(duplicate_range(&bounds, n)).to_owned();
        let envelope: Vec<_> = self.envelope.iter().map(|e| e.slice(duplicate_range(&bounds, n)).to_owned()).collect();

        Self::new(inner, self.score, envelope.into_boxed_slice())
    }
}

fn duplicate_range(bounds: &impl std::ops::RangeBounds<usize>, len: usize) -> std::ops::Range<usize> {
    match (bounds.start_bound(), bounds.end_bound()) {
        (std::ops::Bound::Included(i), std::ops::Bound::Included(j)) => *i..(*j + 1),
        (std::ops::Bound::Included(i), std::ops::Bound::Excluded(j)) => *i..*j,
        (std::ops::Bound::Included(i), std::ops::Bound::Unbounded) => *i..len,
        (std::ops::Bound::Excluded(i), std::ops::Bound::Included(j)) => *i..(*j + 1),
        (std::ops::Bound::Excluded(i), std::ops::Bound::Excluded(j)) => *i..*j,
        (std::ops::Bound::Excluded(i), std::ops::Bound::Unbounded) => *i..len,
        (std::ops::Bound::Unbounded, std::ops::Bound::Included(j)) => 0..(*j + 1),
        (std::ops::Bound::Unbounded, std::ops::Bound::Excluded(j)) => 0..*j,
        (std::ops::Bound::Unbounded, std::ops::Bound::Unbounded) => 0..len,
    }
}