use itertools::multizip;
use mzdeisotope::{scorer::ScoreType, solution::DeconvolvedSolutionPeak};
use num_traits::Zero;
use std::{
    boxed::Box,
    cmp::Ordering,
    ops::{Bound, RangeBounds},
};
use tracing::warn;

use mzpeaks::{
    feature::{ChargedFeature, FeatureView, TimeArray},
    feature_map::FeatureMap,
    peak::MZPoint,
    prelude::*,
    IonMobility, Mass, MZ,
};

use mzsignal::feature_mapping::graph::{
    ChargeAwareFeatureMerger, FeatureGraphBuilder, FeatureNode,
};

use mzdata::{
    prelude::*,
    spectrum::{BinaryArrayMap, BinaryDataArrayType},
    utils::mass_charge_ratio,
};
use mzdata::{
    spectrum::bindata::{
        ArrayRetrievalError, ArrayType, BinaryArrayMap3D, BuildArrayMap3DFrom, BuildFromArrayMap3D,
        DataArray,
    },
    utils::neutral_mass,
};

#[derive(Default, Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    pub fn iter(&self) -> std::iter::Zip<std::slice::Iter<'_, f64>, std::slice::Iter<'_, f32>> {
        self.mz.iter().zip(self.intensity.iter())
    }

    pub fn as_feature_view<Y>(&'a self, time: &'a [f64]) -> FeatureView<'a, MZ, Y> {
        let start = 0;
        let end = time.len().min(self.len());

        FeatureView::new(&self.mz[start..end], &time, &self.intensity[start..end])
    }
}

#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeconvolvedSolutionFeature<Y: Clone> {
    inner: ChargedFeature<Mass, Y>,
    pub score: ScoreType,
    pub scores: Vec<ScoreType>,
    envelope: Box<[MZPointSeries]>,
}

impl<Y: Clone> KnownChargeMut for DeconvolvedSolutionFeature<Y> {
    fn charge_mut(&mut self) -> &mut i32 {
        &mut self.inner.charge
    }
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

    pub fn as_inner(&self) -> &ChargedFeature<Mass, Y> {
        &self.inner
    }

    pub fn into_inner(
        self,
    ) -> (
        ChargedFeature<Mass, Y>,
        Vec<ScoreType>,
        Box<[MZPointSeries]>,
    ) {
        (self.inner, self.scores, self.envelope)
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
        let n_before = self.inner.len();
        self.inner.push_raw(peak.neutral_mass, time, peak.intensity);
        let did_resize = self.len() != n_before;
        if did_resize {
            self.scores.push(peak.score);
            if self.envelope.is_empty() {
                let mut env_set = Vec::new();
                for pt in peak.envelope.iter() {
                    let mut series = MZPointSeries::default();
                    series.push(pt);
                    env_set.push(series);
                }
                self.envelope = env_set.into_boxed_slice();
            } else {
                self.envelope
                    .iter_mut()
                    .zip(peak.envelope.iter())
                    .for_each(|(ev, pt)| ev.push(pt));
            }
        } else {
            let q = self.scores.last_mut().unwrap();
            *q = peak.score.max(*q);
            self.envelope
                .iter_mut()
                .zip(peak.envelope.iter())
                .for_each(|(ev, pt)| {
                    let last_int = *ev.intensity.last().unwrap();
                    let int = last_int + pt.intensity;
                    let last_mz = *ev.mz.last().unwrap();
                    let mz =
                        ((pt.mz * pt.intensity as f64) + (last_mz * last_int as f64)) / int as f64;
                    *ev.mz.last_mut().unwrap() = mz;
                    *ev.intensity.last_mut().unwrap() = int
                });
        }
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
    fn clear(&mut self) {
        self.inner.clear();
        for e in self.envelope.iter_mut() {
            e.intensity.clear();
            e.mz.clear();
        }
    }

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

    fn intensity_view(&self) -> &[f32] {
        self.inner.intensity_view()
    }
}

impl<Y0: Clone> FeatureLike<Mass, Y0> for DeconvolvedSolutionFeature<Y0> {
    fn len(&self) -> usize {
        <ChargedFeature<Mass, Y0> as FeatureLike<Mass, Y0>>::len(&self.inner)
    }

    fn iter(&self) -> impl Iterator<Item = (f64, f64, f32)> {
        <ChargedFeature<Mass, Y0> as FeatureLike<Mass, Y0>>::iter(&self.inner)
    }
}

impl<Y: Clone> PartialOrd for DeconvolvedSolutionFeature<Y> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            return Some(Ordering::Equal);
        }
        match self.neutral_mass().total_cmp(&other.neutral_mass()) {
            Ordering::Equal => {}
            x => return Some(x),
        };
        self.start_time()
            .partial_cmp(&other.start_time())
            .map(|c| c.then(self.score.total_cmp(&other.score)))
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

impl<Y0: Clone> CoordinateLike<MZ> for DeconvolvedSolutionFeature<Y0> {
    fn coordinate(&self) -> f64 {
        <ChargedFeature<Mass, Y0> as CoordinateLike<MZ>>::coordinate(&self.inner)
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

// Collapse features that somehow have repeated time points
pub(crate) fn reflow_feature<Y: Clone + Default>(feature: DeconvolvedSolutionFeature<Y>) -> DeconvolvedSolutionFeature<Y> {
    let mut sink = DeconvolvedSolutionFeature::default();
    *sink.charge_mut() = feature.charge();
    sink.score = feature.score;

    for (peak, time) in feature.iter_peaks() {
        sink.push_peak(&peak, time);
    }
    sink
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
    ) -> Vec<FeatureNode> {
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
            let mut acc: DeconvolvedSolutionFeature<Y> = DeconvolvedSolutionFeature::default();
            acc.inner.charge = features_of[0].charge();
            for f in features_of.iter() {
                acc.score = acc.score.max(f.score);
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

const DECONVOLUTION_SCORE_ARRAY_NAME: &str = "deconvolution score array";
const SUMMARY_SCORE_ARRAY_NAME: &str = "summary deconvolution score array";
const ISOTOPIC_ENVELOPE_ARRAY_NAME: &str = "isotopic envelopes array";
const FEATURE_IDENTIFIER_ARRAY_NAME: &str = "feature identifier array";

impl BuildArrayMapFrom for DeconvolvedSolutionFeature<IonMobility> {
    fn arrays_included(&self) -> Option<Vec<ArrayType>> {
        Some(vec![
            ArrayType::MZArray,
            ArrayType::IntensityArray,
            ArrayType::ChargeArray,
            ArrayType::DeconvolutedIonMobilityArray,
            ArrayType::nonstandard(SUMMARY_SCORE_ARRAY_NAME),
            ArrayType::nonstandard(DECONVOLUTION_SCORE_ARRAY_NAME),
            ArrayType::nonstandard(FEATURE_IDENTIFIER_ARRAY_NAME),
        ])
    }

    fn as_arrays(source: &[Self]) -> BinaryArrayMap {
        let m = source.len();
        let n: usize = source.iter().map(|f| f.len()).sum();
        let n_envelope = source
            .iter()
            .map(|f| f.envelope.iter().map(|e| e.len() + 2).sum::<usize>())
            .sum::<usize>()
            * 2;

        let mut mz_array: Vec<u8> = Vec::with_capacity(n * BinaryDataArrayType::Float64.size_of());

        let mut intensity_array: Vec<u8> =
            Vec::with_capacity(n * BinaryDataArrayType::Float32.size_of());

        let mut ion_mobility_array: Vec<u8> =
            Vec::with_capacity(n * BinaryDataArrayType::Float64.size_of());

        let mut charge_array: Vec<u8> =
            Vec::with_capacity(n * BinaryDataArrayType::Int32.size_of());

        let mut score_array: Vec<u8> =
            Vec::with_capacity(n * BinaryDataArrayType::Float32.size_of());

        let mut summary_score_array: Vec<u8> =
            Vec::with_capacity(m * BinaryDataArrayType::Float32.size_of());

        let mut marker_array: Vec<u8> =
            Vec::with_capacity(n * BinaryDataArrayType::Int32.size_of());

        let mut isotopic_envelope_array: Vec<u8> =
            Vec::with_capacity(n_envelope * BinaryDataArrayType::Float32.size_of());

        let mut acc = Vec::with_capacity(n);
        source.iter().enumerate().for_each(|(i, f)| {
            let envelope = &f.envelope;
            let n_env = envelope.len() as f32;
            let m_env = envelope.first().map(|f| f.len()).unwrap_or_default() as f32;

            isotopic_envelope_array.extend_from_slice(&n_env.to_le_bytes());
            isotopic_envelope_array.extend_from_slice(&m_env.to_le_bytes());
            for env in envelope.iter() {
                for (x, y) in env.iter() {
                    isotopic_envelope_array.extend_from_slice(&((*x) as f32).to_le_bytes());
                    isotopic_envelope_array.extend_from_slice(&(*y).to_le_bytes());
                }
                isotopic_envelope_array.extend_from_slice(&(0.0f32).to_le_bytes());
                isotopic_envelope_array.extend_from_slice(&(0.0f32).to_le_bytes());
            }

            summary_score_array.extend(f.score.to_le_bytes());
            f.iter().enumerate().for_each(|(j, (mass, im, inten))| {
                acc.push((
                    mass_charge_ratio(mass, f.charge()),
                    im,
                    inten,
                    f.charge(),
                    f.scores[j],
                    i,
                ))
            })
        });

        acc.sort_by(
            |(mz_a, im_a, _, _, _, key_a), (mz_b, im_b, _, _, _, key_b)| {
                mz_a.total_cmp(mz_b)
                    .then(im_a.total_cmp(im_b))
                    .then(key_a.cmp(key_b))
            },
        );

        for (mz, im, inten, charge, score, key) in acc.iter() {
            mz_array.extend(mz.to_le_bytes());
            intensity_array.extend(inten.to_le_bytes());
            ion_mobility_array.extend(im.to_le_bytes());
            charge_array.extend(charge.to_le_bytes());
            score_array.extend(score.to_le_bytes());
            marker_array.extend((*key as i32).to_le_bytes());
        }

        let mut map = BinaryArrayMap::default();
        map.add(DataArray::wrap(
            &ArrayType::MZArray,
            BinaryDataArrayType::Float64,
            mz_array,
        ));
        map.add(DataArray::wrap(
            &ArrayType::IntensityArray,
            BinaryDataArrayType::Float32,
            intensity_array,
        ));
        map.add(DataArray::wrap(
            &ArrayType::ChargeArray,
            BinaryDataArrayType::Int32,
            charge_array,
        ));
        map.add(DataArray::wrap(
            &ArrayType::DeconvolutedIonMobilityArray,
            BinaryDataArrayType::Float64,
            ion_mobility_array,
        ));
        map.add(DataArray::wrap(
            &ArrayType::nonstandard(SUMMARY_SCORE_ARRAY_NAME),
            BinaryDataArrayType::Float32,
            summary_score_array,
        ));
        map.add(DataArray::wrap(
            &ArrayType::nonstandard(DECONVOLUTION_SCORE_ARRAY_NAME),
            BinaryDataArrayType::Float32,
            score_array,
        ));
        map.add(DataArray::wrap(
            &ArrayType::nonstandard(FEATURE_IDENTIFIER_ARRAY_NAME),
            BinaryDataArrayType::Int32,
            marker_array,
        ));
        map.add(DataArray::wrap(
            &ArrayType::nonstandard(ISOTOPIC_ENVELOPE_ARRAY_NAME),
            BinaryDataArrayType::Float32,
            isotopic_envelope_array,
        ));

        map
    }
}

impl BuildArrayMap3DFrom for DeconvolvedSolutionFeature<IonMobility> {}

impl BuildFromArrayMap for DeconvolvedSolutionFeature<IonMobility> {
    fn arrays_required() -> Option<Vec<ArrayType>> {
        Some(vec![
            ArrayType::MZArray,
            ArrayType::IntensityArray,
            ArrayType::ChargeArray,
            ArrayType::DeconvolutedIonMobilityArray,
            ArrayType::nonstandard(SUMMARY_SCORE_ARRAY_NAME),
            ArrayType::nonstandard(DECONVOLUTION_SCORE_ARRAY_NAME),
            ArrayType::nonstandard(FEATURE_IDENTIFIER_ARRAY_NAME),
        ])
    }

    fn try_from_arrays(arrays: &BinaryArrayMap) -> Result<Vec<Self>, ArrayRetrievalError> {
        let arrays_3d = arrays.try_into()?;
        Self::try_from_arrays_3d(&arrays_3d)
    }
}

impl BuildFromArrayMap3D for DeconvolvedSolutionFeature<IonMobility> {
    fn try_from_arrays_3d(arrays: &BinaryArrayMap3D) -> Result<Vec<Self>, ArrayRetrievalError> {
        let key = ArrayType::nonstandard(FEATURE_IDENTIFIER_ARRAY_NAME);
        let mut n: usize = 0;
        for (_, arr) in arrays.iter() {
            if arr.is_empty() {
                continue;
            }
            if let Some(arr) = arr.get(&key) {
                if let Some(i) = arr.iter_i32()?.map(|i| i as usize).max() {
                    n = n.max(i);
                }
            }
        }

        let isotopic_envelope_array_key = ArrayType::nonstandard(ISOTOPIC_ENVELOPE_ARRAY_NAME);
        let score_array_key = ArrayType::nonstandard(DECONVOLUTION_SCORE_ARRAY_NAME);
        let summary_score_array_key = ArrayType::nonstandard(SUMMARY_SCORE_ARRAY_NAME);

        if n == 0 {
            return Ok(Vec::new());
        }

        let mut index = Vec::with_capacity(n + 1);
        index.resize(n + 1, Self::default());

        for (im, arr) in arrays.iter() {
            if arr.is_empty() {
                continue;
            }

            let mz_array = arr.mzs()?;
            let intensity_array = arr.intensities()?;
            let charge_array = arr.charges()?;
            let scores_array = arr
                .get(&score_array_key)
                .ok_or_else(|| ArrayRetrievalError::NotFound(score_array_key.clone()))?
                .to_f32()?;
            let marker_array = arr
                .get(&key)
                .ok_or_else(|| ArrayRetrievalError::NotFound(key.clone()))?
                .to_i32()?;

            for (mz, inten, charge, score, key_i) in multizip((
                mz_array.iter(),
                intensity_array.iter(),
                charge_array.iter(),
                scores_array.iter(),
                marker_array.iter(),
            )) {
                let f = &mut index[(*key_i) as usize];
                if f.is_empty() {
                    f.inner.charge = *charge;
                }
                f.score += *score;
                f.push_raw(neutral_mass(*mz, *charge), im, *inten);
                f.scores.push(*score);
            }
        }

        if let Some(isotopic_envelopes_array) =
            arrays.additional_arrays.get(&isotopic_envelope_array_key)
        {
            let mut isotopic_envelopes = Vec::new();
            let isotopic_envelopes_array = isotopic_envelopes_array.to_f32()?;
            let mut chunks = isotopic_envelopes_array.chunks_exact(2);
            while let Some((n_traces, _trace_size)) = chunks.next().map(|header| {
                let n_traces = header[0];
                let trace_size = header[1];
                (n_traces as usize, trace_size as usize)
            }) {
                let mut traces: Vec<MZPointSeries> = Vec::with_capacity(n_traces);
                let mut current_trace = MZPointSeries::default();
                while traces.len() < n_traces {
                    while let Some(block) = chunks.next() {
                        let mz = block[0] as f64;
                        let intensity = block[1];
                        if mz.is_zero() && intensity.is_zero() {
                            break;
                        }
                        current_trace.push(MZPoint::new(mz, intensity));
                    }
                    if current_trace.len() == 0 {
                        warn!("Empty trace detected");
                    }
                    traces.push(current_trace);
                    current_trace = MZPointSeries::default();
                }
                isotopic_envelopes.push(Some(traces));
            }
            for (i, f) in index.iter_mut().enumerate() {
                f.envelope = isotopic_envelopes[i].take().unwrap().into_boxed_slice();
            }
        }

        if let Some(summary_score_array) = arrays.additional_arrays.get(&summary_score_array_key) {
            let summary_scores = summary_score_array.to_f32()?;
            for (score, f) in summary_scores.iter().zip(index.iter_mut()) {
                f.score = *score;
            }
        }
        Ok(index)
    }
}
