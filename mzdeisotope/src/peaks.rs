//! Helper types for writing exhaustive deconvolution algorithms.

use chemical_elements::isotopic_pattern::TheoreticalIsotopicPattern;
use mzpeaks::{
    coordinate::{SimpleInterval, Span1D},
    prelude::*,
    CentroidPeak, IndexType, MZPeakSetType,
};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hash;
use std::mem;
use std::ops::{Index, Range};

use identity_hash::{BuildIdentityHasher, IdentityHashable};

use crate::charge::{quick_charge_w, ChargeListIter, ChargeRange};
use crate::isotopic_fit::IsotopicFit;

const PEAK_ELIMINATION_FACTOR: f32 = 0.7;

/// An integral type for storing a transformed m/z that is sufficiently unique for hashing.
/// Good for m/z values up to 2_147_483.647. This should be just fine.
pub type Placeholder = i32;

/// A combination of traits that [`WorkingPeakSet`] needs to function.
pub trait PeakLike: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut {}

impl<T> PeakLike for T where T: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut {}

/// Represent a sufficiently unique key for indexing or hashing
/// peaks from a peak list, or denoting a theoretical but absent
/// peak.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PeakKey {
    /// A matched experimental peak index
    Matched(u32),
    /// A synthetic marker for a peak m/z that was not found but
    /// which may anchor a solution containing real peaks.
    Placeholder(Placeholder),
}

impl Hash for PeakKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            PeakKey::Matched(x) => state.write_u32(*x),
            PeakKey::Placeholder(x) => state.write_i32(*x),
        }
    }
}

impl IdentityHashable for PeakKey {}

impl PeakKey {
    /// Check if the key corresponds to a matched peak
    pub fn is_matched(&self) -> bool {
        matches!(self, Self::Matched(_))
    }

    /// Check if the key corresponds to a placeholder
    pub fn is_placeholder(&self) -> bool {
        matches!(self, Self::Placeholder(_))
    }

    pub(crate) fn to_index_unchecked(self) -> u32 {
        match self {
            PeakKey::Matched(i) => i,
            PeakKey::Placeholder(_) => panic!("PeakKey index requested, but found a placeholder"),
        }
    }
}

/// A cache mapping [`Placeholder`] values to concrete [`PeakLike`] values for a single
/// peak list deconvolution problem.
#[derive(Debug)]
pub struct PlaceholderCache<C: PeakLike> {
    placeholders: HashMap<Placeholder, C, BuildIdentityHasher<Placeholder>>,
}

impl<C: PeakLike> Default for PlaceholderCache<C> {
    fn default() -> Self {
        Self {
            placeholders: Default::default(),
        }
    }
}

/// Helper trait for producing [`Placeholder`] values for [`PeakKey`] instances
pub trait MZCaching {
    fn key_for(&self, mz: f64) -> Placeholder {
        (mz * 1000.0).round() as Placeholder
    }
}

impl<C: PeakLike> PlaceholderCache<C> {
    /// Create a [`Placeholder`] value for a given m/z and store
    /// an associated [`PeakLike`] instance in the cache with
    /// it.
    pub fn create(&mut self, mz: f64) -> Placeholder {
        let key = self.key_for(mz);
        self.placeholders
            .entry(key)
            .or_insert_with(|| CentroidPeak::new(mz, 1.0, IndexType::MAX).into());
        key
    }

    /// As [`Self::create`] but instead return the peak, not the key
    pub fn create_and_get(&mut self, mz: f64) -> &C {
        let key = self.key_for(mz);
        self.create(mz);
        self.placeholders.get(&key).unwrap()
    }

    /// Map a specified m/z to a pre-existing peak
    pub fn get(&self, mz: f64) -> Option<&C> {
        let key = self.key_for(mz);
        self.placeholders.get(&key)
    }

    /// Map a specified [`Placeholder`] key to a pre-existing peak
    pub fn get_key(&self, key: &Placeholder) -> &C {
        self.placeholders.get(key).unwrap()
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.placeholders.clear()
    }
}

impl<C: PeakLike> MZCaching for PlaceholderCache<C> {}

/// A cache mapping a pair of [`Placeholder`] to real indices in an
/// experimental peak list
#[derive(Debug, Clone, Default)]
pub struct SliceCache {
    range_to_indices: HashMap<(Placeholder, Placeholder), Range<usize>>,
}

impl MZCaching for SliceCache {}

impl SliceCache {
    pub fn clear(&mut self) {
        self.range_to_indices.clear()
    }

    pub fn entry(&mut self, m1: f64, m2: f64) -> Entry<(Placeholder, Placeholder), Range<usize>> {
        let key = self.key_from(m1, m2);
        self.range_to_indices.entry(key)
    }

    pub fn key_from(&mut self, m1: f64, m2: f64) -> (Placeholder, Placeholder) {
        let i1 = self.key_for(m1);
        let i2 = self.key_for(m2);
        (i1, i2)
    }
}

/// A wrapper enclosing an [`MZPeakSetType`] with sets of caches that make
/// writing deconvoluters more convenient.
#[derive(Debug)]
pub struct WorkingPeakSet<C: PeakLike + IntensityMeasurementMut> {
    pub peaks: MZPeakSetType<C>,
    pub placeholders: PlaceholderCache<C>,
    pub slice_cache: SliceCache,
}

impl<C: PeakLike + IntensityMeasurementMut> WorkingPeakSet<C> {
    pub fn new(peaks: MZPeakSetType<C>) -> Self {
        Self {
            peaks,
            placeholders: PlaceholderCache::default(),
            slice_cache: SliceCache::default(),
        }
    }

    pub fn tic(&self) -> f32 {
        self.peaks.iter().map(|p| p.intensity()).sum()
    }

    pub fn between(&mut self, m1: f64, m2: f64) -> Range<usize> {
        match self.slice_cache.entry(m1, m2) {
            Entry::Occupied(r) => r.get().clone(),
            Entry::Vacant(v) => {
                let ivs = self.peaks.between(m1, m2, Tolerance::Da(0.001));
                let r = if ivs.is_empty() {
                    0..0
                } else {
                    let start = ivs.first().unwrap().get_index() as usize;
                    let end = (ivs.last().unwrap().get_index() + 1) as usize;
                    start..end
                };
                v.insert(r.clone());
                r
            }
        }
    }

    pub fn match_theoretical(
        &mut self,
        tid: &TheoreticalIsotopicPattern,
        error_tolerance: Tolerance,
    ) -> (Vec<PeakKey>, usize) {
        let mut peaks = Vec::with_capacity(tid.len());
        let mut missed = 0;
        for peak in tid.iter() {
            let (key, missed_peak) = self.has_peak(peak.mz(), error_tolerance);
            peaks.push(key);
            missed += missed_peak as usize;
        }
        (peaks, missed)
    }

    pub fn has_peak(&mut self, mz: f64, error_tolerance: Tolerance) -> (PeakKey, bool) {
        match self.peaks.has_peak(mz, error_tolerance) {
            Some(peak) => (PeakKey::Matched(peak.get_index()), false),
            None => (PeakKey::Placeholder(self.placeholders.create(mz)), true),
        }
    }

    pub fn has_peak_direct(&self, mz: f64, error_tolerance: Tolerance) -> Option<&C> {
        self.peaks.has_peak(mz, error_tolerance)
    }

    pub fn collect_for(&self, keys: &[PeakKey]) -> Vec<&C> {
        let mut result = Vec::with_capacity(keys.len());
        for key in keys.iter() {
            result.push(self.get(key))
        }
        result
    }

    pub fn get(&self, key: &PeakKey) -> &C {
        match key {
            PeakKey::Matched(i) => &self.peaks[*i as usize],
            PeakKey::Placeholder(i) => self.placeholders.get_key(i),
        }
    }

    pub fn get_mut(&mut self, key: &PeakKey) -> Option<&mut C> {
        match key {
            PeakKey::Matched(i) => Some(&mut self.peaks[*i as usize]),
            PeakKey::Placeholder(_) => None,
        }
    }

    pub fn len(&self) -> usize {
        self.peaks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.peaks.is_empty()
    }

    pub fn key_iter(&self) -> PeakKeyIter {
        PeakKeyIter::descending(self.peaks.len())
    }

    /// Find all peaks in the specified m/z `intervals` and set their intensities
    /// to `1.0`, making them skippable, and of equal value to placeholder peaks.
    pub fn mask_peaks_in_intervals(&mut self, intervals: &[SimpleInterval<f64>]) -> u32 {
        let mut masked_counter = 0;
        for iv in intervals {
            let peaks_in = self.peaks.between(iv.start, iv.end, Tolerance::Da(0.001));
            let start_i = peaks_in
                .first()
                .map(|p| p.get_index() as usize)
                .unwrap_or_default();
            let end_i = peaks_in
                .last()
                .map(|p| p.get_index() as usize + 1)
                .unwrap_or_default();
            let peaks = self.peaks.as_mut_slice();
            for i in start_i..end_i {
                if let Some(p) = peaks.get_mut(i) {
                    if iv.contains(&p.mz()) {
                        masked_counter += 1;
                        *p.intensity_mut() = 1.0;
                    }
                }
            }
        }
        masked_counter
    }

    pub fn subtract_theoretical_intensity(&mut self, fit: &IsotopicFit) {
        fit.experimental
            .iter()
            .zip(fit.theoretical.iter())
            .for_each(|(e, t)| {
                if let Some(peak) = self.get_mut(e) {
                    let threshold = peak.intensity() * PEAK_ELIMINATION_FACTOR;
                    let new = peak.intensity() - t.intensity();
                    if (new - threshold).abs() < 1e-3 || new < 0.0 {
                        *peak.intensity_mut() = 1.0;
                    } else {
                        *peak.intensity_mut() = new;
                    }
                }
            });
    }

    pub fn quick_charge(&self, position: usize, charge_range: ChargeRange) -> ChargeListIter {
        quick_charge_w(&self.peaks[0..], position, charge_range)
    }

    pub fn clear(&mut self) {
        self.slice_cache.clear();
        self.placeholders.clear();
    }

    pub fn set_peaks(&mut self, mut peaks: MZPeakSetType<C>) -> MZPeakSetType<C> {
        mem::swap(&mut self.peaks, &mut peaks);
        self.clear();
        peaks
    }

    pub fn as_slice(&self) -> &[C] {
        self.peaks.as_slice()
    }

    /// Find all peaks that are not part of the isotopic pattern fits in `fits` to find
    /// the m/z intervals where nothing is a viable isotopic pattern.
    ///
    /// The intervals must be at least `3 * min_width` m/z wide to be reported.
    pub fn find_unused_peaks(&self, fits: &[IsotopicFit], min_width: f64) -> Vec<SimpleInterval<f64>> {
        // Build a peak index mask that is true if a peak is used in a fit, false otherwise
        let mut mask = Vec::with_capacity(self.len());
        mask.resize(self.len(), false);
        for fit in fits {
            for key in fit.experimental.iter() {
                if let PeakKey::Matched(i) = key {
                    mask[*i as usize] = true;
                }
            }
        }

        // Build series of m/z intervals spanning dead zones in the peak mask
        let mut spans: Vec<SimpleInterval<f64>> = Vec::new();
        let mut last_mz = Some(0.0);
        for (peak, has_fit) in self.peaks.iter().zip(mask.into_iter()) {
            if has_fit {
                if last_mz.is_some() {
                    spans.push(SimpleInterval::new(last_mz.unwrap(), peak.mz()));
                    last_mz = None
                }
            } else {
                if last_mz.is_none() {
                    last_mz = Some(peak.mz())
                }
            }
        }

        if last_mz.is_some() {
            spans.push(SimpleInterval::new(last_mz.unwrap(), f64::INFINITY));
        }

        // Truncate intervals to enforce the minimum width criterion and to "unpad" them
        // to guarantee there is no spillage from adjacent regions.
        let spans: Vec<_> = spans
            .into_iter()
            .filter_map(|mut iv| {
                iv.start = iv.start + min_width;
                iv.end = iv.end - min_width;
                if (iv.end - iv.start) < min_width {
                    None
                } else {
                    Some(iv)
                }
            })
            .collect();
        spans
    }
}

impl<C: PeakLike + IntensityMeasurementMut> Index<usize> for WorkingPeakSet<C> {
    type Output = C;

    fn index(&self, index: usize) -> &Self::Output {
        &self.peaks[index]
    }
}

impl<C: PeakLike + IntensityMeasurementMut> Index<PeakKey> for WorkingPeakSet<C> {
    type Output = C;

    fn index(&self, index: PeakKey) -> &Self::Output {
        match index {
            PeakKey::Matched(i) => self.peaks.index(i as usize),
            PeakKey::Placeholder(i) => self.placeholders.get_key(&i),
        }
    }
}

/// An iterator over a range of [`PeakKey`]s corresponding to [`PeakKey::Matched`]
/// instances.
#[derive(Debug, Clone)]
pub struct PeakKeyIter {
    total: usize,
    current: usize,
    sign: i8,
}

impl PeakKeyIter {
    pub fn new(total: usize, current: usize, sign: i8) -> Self {
        Self {
            total,
            current,
            sign,
        }
    }

    pub fn descending(total: usize) -> Self {
        let current = if total == 0 { usize::MAX } else { total - 1 };

        Self {
            total,
            current,
            sign: -1,
        }
    }

    pub fn next_key(&mut self) -> Option<PeakKey> {
        if self.sign > 0 {
            if self.current < self.total - 1 {
                let res = PeakKey::Matched(self.current as u32);
                self.current += 1;
                Some(res)
            } else {
                None
            }
        } else if self.current > 0 && self.current < self.total {
            let res = PeakKey::Matched(self.current as u32);
            self.current -= 1;
            Some(res)
        } else if self.current == 0 {
            let res = PeakKey::Matched(self.current as u32);
            self.current = usize::MAX;
            Some(res)
        } else {
            None
        }
    }
}

impl Iterator for PeakKeyIter {
    type Item = PeakKey;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_key()
    }
}
