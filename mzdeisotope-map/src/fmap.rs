use std::ops::{Index, IndexMut};

use mzpeaks::{
    feature::Feature,
    feature_map::FeatureMap,
    prelude::*,
    MZ,
};

#[derive(Debug, Default, Clone)]
pub struct IndexedFeature<Y> {
    pub(crate) feature: Feature<MZ, Y>,
    pub mz: f64,
    pub intensity: f32,
    pub charges: Option<Box<[i32]>>,
}

impl<Y> IndexedFeature<Y> {
    pub fn invalidate(&mut self) {
        self.mz = self.feature.mz();
        self.intensity = self.feature.intensity();
        self.charges = None;
    }

    pub fn charges(&self) -> Option<&[i32]> {
        self.charges.as_deref()
    }

    pub fn charges_mut(&mut self) -> &mut Option<Box<[i32]>> {
        &mut self.charges
    }
}

impl<Y> IndexedFeature<Y> {
    pub fn total_intensity(&self) -> f32 {
        self.intensity
    }
}

impl<Y> AsRef<Feature<MZ, Y>> for IndexedFeature<Y> {
    fn as_ref(&self) -> &Feature<MZ, Y> {
        &self.feature
    }
}

impl<Y> From<Feature<MZ, Y>> for IndexedFeature<Y> {
    fn from(value: Feature<MZ, Y>) -> Self {
        let mz = value.coordinate();
        let intensity = value.intensity();
        Self {
            feature: value,
            mz,
            intensity,
            charges: None,
        }
    }
}

impl<Y> From<IndexedFeature<Y>> for Feature<MZ, Y> {
    fn from(value: IndexedFeature<Y>) -> Self {
        value.feature
    }
}

impl<Y> PartialOrd for IndexedFeature<Y> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.feature.partial_cmp(&other.feature)
    }
}

impl<Y> PartialEq for IndexedFeature<Y> {
    fn eq(&self, other: &Self) -> bool {
        self.feature == other.feature
    }
}

impl<Y1> FeatureLike<MZ, Y1> for IndexedFeature<Y1> {
    fn len(&self) -> usize {
        <Feature<MZ, Y1> as FeatureLike<MZ, Y1>>::len(&self.feature)
    }

    fn iter(&self) -> impl Iterator<Item = (f64, f64, f32)> {
        <Feature<MZ, Y1> as FeatureLike<MZ, Y1>>::iter(&self.feature)
    }
}

impl<Y1> IntensityMeasurement for IndexedFeature<Y1> {
    fn intensity(&self) -> f32 {
        self.intensity
    }
}

impl<Y1> FeatureLikeMut<MZ, Y1> for IndexedFeature<Y1> {
    fn iter_mut(&mut self) -> impl Iterator<Item = (&mut f64, &mut f64, &mut f32)> {
        <Feature<MZ, Y1> as FeatureLikeMut<MZ, Y1>>::iter_mut(&mut self.feature)
    }

    fn push<T: CoordinateLike<MZ> + IntensityMeasurement>(&mut self, pt: &T, time: f64) {
        <Feature<MZ, Y1> as FeatureLikeMut<MZ, Y1>>::push(&mut self.feature, pt, time)
    }

    fn push_raw(&mut self, x: f64, y: f64, z: f32) {
        <Feature<MZ, Y1> as FeatureLikeMut<MZ, Y1>>::push_raw(&mut self.feature, x, y, z)
    }

    fn clear(&mut self) {
        <Feature<MZ, Y1> as FeatureLikeMut<MZ, Y1>>::clear(&mut self.feature)
    }
}

impl<T: CoordinateLike<MZ> + IntensityMeasurement, Y1> BuildFromPeak<T> for IndexedFeature<Y1> {
    fn push_peak(&mut self, value: &T, time: f64) {
        <Feature<MZ, Y1> as BuildFromPeak<T>>::push_peak(&mut self.feature, value, time)
    }
}

impl<Y1> TimeInterval<Y1> for IndexedFeature<Y1> {
    fn apex_time(&self) -> Option<f64> {
        <Feature<MZ, Y1> as TimeInterval<Y1>>::apex_time(&self.feature)
    }

    fn area(&self) -> f32 {
        <Feature<MZ, Y1> as TimeInterval<Y1>>::area(&self.feature)
    }

    fn end_time(&self) -> Option<f64> {
        <Feature<MZ, Y1> as TimeInterval<Y1>>::end_time(&self.feature)
    }

    fn start_time(&self) -> Option<f64> {
        <Feature<MZ, Y1> as TimeInterval<Y1>>::start_time(&self.feature)
    }

    fn iter_time(&self) -> impl Iterator<Item = f64> {
        <Feature<MZ, Y1> as TimeInterval<Y1>>::iter_time(&self.feature)
    }

    fn find_time(&self, time: f64) -> (Option<usize>, f64) {
        <Feature<MZ, Y1> as TimeInterval<Y1>>::find_time(&self.feature, time)
    }
}

impl<Y1> CoordinateLike<MZ> for IndexedFeature<Y1> {
    fn coordinate(&self) -> f64 {
        self.mz
    }
}

impl<Y1> TimeArray<Y1> for IndexedFeature<Y1> {
    fn time_view(&self) -> &[f64] {
        <Feature<MZ, Y1> as TimeArray<Y1>>::time_view(&self.feature)
    }

    fn intensity_view(&self) -> &[f32] {
        <Feature<MZ, Y1> as TimeArray<Y1>>::intensity_view(&self.feature)
    }
}

#[derive(Debug, Default, Clone)]
pub struct IndexedFeatureMap<Y> {
    features: FeatureMap<MZ, Y, IndexedFeature<Y>>,
    index: Vec<f64>,
}

impl<Y> AsRef<FeatureMap<MZ, Y, IndexedFeature<Y>>> for IndexedFeatureMap<Y> {
    fn as_ref(&self) -> &FeatureMap<MZ, Y, IndexedFeature<Y>> {
        &self.features
    }
}

impl<Y> FromIterator<IndexedFeature<Y>> for IndexedFeatureMap<Y> {
    fn from_iter<T: IntoIterator<Item = IndexedFeature<Y>>>(iter: T) -> Self {
        FeatureMap::from_iter(iter.into_iter(), true).into()
    }
}

impl<Y> From<FeatureMap<MZ, Y, IndexedFeature<Y>>> for IndexedFeatureMap<Y> {
    fn from(value: FeatureMap<MZ, Y, IndexedFeature<Y>>) -> Self {
        let mut this = Self {
            features: value,
            index: Vec::new(),
        };
        this.build_index();
        this
    }
}

impl<Y> From<FeatureMap<MZ, Y, Feature<MZ, Y>>> for IndexedFeatureMap<Y> {
    fn from(value: FeatureMap<MZ, Y, Feature<MZ, Y>>) -> Self {
        let mut this = Self {
            features: value.into_iter().map(IndexedFeature::from).collect(),
            index: Vec::new(),
        };
        this.build_index();
        this
    }
}

impl<Y> IndexedFeatureMap<Y> {
    pub fn new(features: Vec<IndexedFeature<Y>>) -> Self {
        let features = FeatureMap::new(features);
        let mut this = Self {
            features,
            index: Vec::new()
        };
        this.build_index();
        this
    }

    fn build_index(&mut self) {
        self.index = self.features.iter().map(|f| f.mz).collect();
    }

    pub fn as_slice(&self) -> &[IndexedFeature<Y>] {
        self.features.as_slice()
    }

    pub fn first(&self) -> Option<&IndexedFeature<Y>> {
        self.features.first()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn into_inner(self) -> FeatureMap<MZ, Y, IndexedFeature<Y>> {
        self.features
    }

    pub fn iter(&self) -> std::slice::Iter<'_, IndexedFeature<Y>> {
        self.features.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, IndexedFeature<Y>> {
        self.features.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn empty() -> Self {
        let inner = FeatureMap::<MZ, Y, Feature<MZ, Y>>::empty();
        inner.into()
    }
}

impl<Y> Index<usize> for IndexedFeatureMap<Y> {
    type Output = IndexedFeature<Y>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.features[index]
    }
}

impl<Y> IndexMut<usize> for IndexedFeatureMap<Y> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.features[index]
    }
}

impl<Y1> FeatureMapLikeMut<MZ, Y1, IndexedFeature<Y1>> for IndexedFeatureMap<Y1> {
    fn push(&mut self, feature: IndexedFeature<Y1>) {
        if self.is_empty() {
            self.index.push(feature.mz);
            self.features.push(feature);
        } else {
            let is_tail =
                self.features.last().as_ref().unwrap().coordinate() <= feature.coordinate();
            self.features.push(feature);
            if !is_tail {
                self.sort();
            }
        }
    }

    fn sort(&mut self) {
        <FeatureMap<MZ, Y1, IndexedFeature<Y1>> as FeatureMapLikeMut<MZ, Y1, IndexedFeature<Y1>>>::sort(&mut self.features);
        self.build_index();
    }
}

impl<Y1> FeatureMapLike<MZ, Y1, IndexedFeature<Y1>> for IndexedFeatureMap<Y1> {
    fn search_by(&self, query: f64) -> Result<usize, usize> {
        self.index.binary_search_by(|b| b.total_cmp(&query))
    }

    fn len(&self) -> usize {
        <FeatureMap<MZ, Y1, IndexedFeature<Y1>> as FeatureMapLike<MZ, Y1, IndexedFeature<Y1>>>::len(
            &self.features,
        )
    }

    fn is_empty(&self) -> bool {
        <FeatureMap<MZ, Y1, IndexedFeature<Y1>> as FeatureMapLike<MZ, Y1, IndexedFeature<Y1>>>::is_empty(&self.features)
    }

    fn get_item(&self, i: usize) -> &IndexedFeature<Y1> {
        <FeatureMap<MZ, Y1, IndexedFeature<Y1>> as FeatureMapLike<MZ, Y1, IndexedFeature<Y1>>>::get_item(&self.features, i)
    }

    unsafe fn get_item_unchecked(&self, i: usize) -> &IndexedFeature<Y1> {
        <FeatureMap<MZ, Y1, IndexedFeature<Y1>> as FeatureMapLike<MZ, Y1, IndexedFeature<Y1>>>::get_item_unchecked(&self.features, i)
    }

    fn get_slice(&self, i: std::ops::Range<usize>) -> &[IndexedFeature<Y1>] {
        <FeatureMap<MZ, Y1, IndexedFeature<Y1>> as FeatureMapLike<MZ, Y1, IndexedFeature<Y1>>>::get_slice(&self.features, i)
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a IndexedFeature<Y1>>
    where
        IndexedFeature<Y1>: 'a,
    {
        <FeatureMap<MZ, Y1, IndexedFeature<Y1>> as FeatureMapLike<MZ, Y1, IndexedFeature<Y1>>>::iter(
            &self.features,
        )
    }
}

pub struct FeatureIter<Y> {
    source: std::vec::IntoIter<IndexedFeature<Y>>,
}

impl<Y> Iterator for FeatureIter<Y> {
    type Item = Feature<MZ, Y>;

    fn next(&mut self) -> Option<Self::Item> {
        self.source.next().map(|f| f.feature)
    }
}

impl<Y> IntoIterator for IndexedFeatureMap<Y> {
    type Item = Feature<MZ, Y>;

    type IntoIter = FeatureIter<Y>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            source: self.features.into_iter(),
        }
    }
}
