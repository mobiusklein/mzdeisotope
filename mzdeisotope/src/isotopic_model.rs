/*! Isotopic models for generating isotopic patterns */
use std::cmp::Ordering;
use std::collections::btree_map::{self, BTreeMap, Entry as BEntry};
#[allow(unused)]
use std::collections::hash_map::{self, Entry, HashMap};
use std::hash;

#[doc(hidden)]
pub use chemical_elements::isotopic_pattern::{
    BafflingRecursiveIsotopicPatternGenerator, TheoreticalIsotopicPattern, Peak as TheorteicalPeak
};

use chemical_elements::{
    neutral_mass, ChemicalComposition, ElementSpecification, PROTON as _PROTON,
};

use mzpeaks::{CentroidLike, IntensityMeasurement};
use num_traits::Float;
use tracing::trace;

pub(crate) fn isclose<T: Float>(a: T, b: T, delta: T) -> bool {
    (a - b).abs() < delta
}

/// The mass of H+, a hydrogen atom minus an electron
pub const PROTON: f64 = _PROTON;

/// A fractional elemental composition with non-ordinal element counts used to represent
/// "averaged" chemical compositions.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct FractionalComposition<'a>(HashMap<ElementSpecification<'a>, f64>);

impl<'a> Extend<(ElementSpecification<'a>, f64)> for FractionalComposition<'a> {
    fn extend<T: IntoIterator<Item = (ElementSpecification<'a>, f64)>>(&mut self, iter: T) {
        <HashMap<ElementSpecification<'a>, f64> as Extend<(ElementSpecification<'a>, f64)>>::extend(
            &mut self.0,
            iter,
        )
    }
}

impl<'a> FromIterator<(ElementSpecification<'a>, f64)> for FractionalComposition<'a> {
    fn from_iter<T: IntoIterator<Item = (ElementSpecification<'a>, f64)>>(iter: T) -> Self {
        let mut this = Self::default();
        this.extend(iter);
        this
    }
}

impl<'a> From<HashMap<ElementSpecification<'a>, f64>> for FractionalComposition<'a> {
    fn from(value: HashMap<ElementSpecification<'a>, f64>) -> Self {
        Self::new(value)
    }
}

impl<'a> From<Vec<(ElementSpecification<'a>, f64)>> for FractionalComposition<'a> {
    fn from(value: Vec<(ElementSpecification<'a>, f64)>) -> Self {
        value.into_iter().collect()
    }
}

impl<'a> FractionalComposition<'a> {
    #[inline]
    pub fn new(composition: HashMap<ElementSpecification<'a>, f64>) -> Self {
        Self(composition)
    }

    #[inline]
    pub fn get<Q>(&self, k: &Q) -> Option<&f64>
    where
        ElementSpecification<'a>: std::borrow::Borrow<Q>,
        Q: hash::Hash + Eq + ?Sized,
    {
        self.0.get(k)
    }

    #[inline]
    pub fn iter(&self) -> hash_map::Iter<'_, ElementSpecification<'a>, f64> {
        self.0.iter()
    }

    #[inline]
    pub fn mass(&self) -> f64 {
        self.iter()
            .map(|(e, c)| e.element.most_abundant_mass * *c)
            .sum()
    }

    pub fn insert(&mut self, k: ElementSpecification<'a>, v: f64) -> Option<f64> {
        self.0.insert(k, v)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn contains_key<Q>(&self, k: &Q) -> bool
    where
        ElementSpecification<'a>: std::borrow::Borrow<Q>,
        Q: hash::Hash + Eq + ?Sized,
    {
        self.0.contains_key(k)
    }

    pub fn iter_mut(&mut self) -> hash_map::IterMut<'_, ElementSpecification<'a>, f64> {
        self.0.iter_mut()
    }
}

/// The capability to generate theoretical isotopic patterns from a given mass
pub trait IsotopicPatternGenerator {
    /// Generate a theoretical isotopic pattern for a given m/z and charge
    ///
    /// # Arguments
    /// - `mz`:  The m/z to use as a point of reference when generating the isotopic pattern
    /// - `charge`: The charge state to use to convert `mz` to a mass
    /// - `charge_carrier`: The mass of the charge carrier, usually a proton
    /// - `truncate_after`: The cumulative abundance percentage of isotopic signal to retain
    /// - `ignore_below`: The minimum abundance percentage of isotopic signal a peak must have to be kept
    fn isotopic_cluster(
        &mut self,
        mz: f64,
        charge: i32,
        charge_carrier: f64,
        truncate_after: f64,
        ignore_below: f64,
    ) -> TheoreticalIsotopicPattern;

    /// Populate a cache, if available, pre-computing isotopic patterns from them and store them
    /// away for future use
    ///
    /// # Arguments
    /// - `min_mz`, `max_mz`: The m/z range to generate patterns between
    /// - `min_charge`, `max_charge`: The charge range to generate patterns over for each m/z
    /// - `charge_carrier`: The mass of the charge carrier, usually a proton
    /// - `truncate_after`: The cumulative abundance percentage of isotopic signal to retain
    /// - `ignore_below`: The minimum abundance percentage of isotopic signal a peak must have to be kept
    #[allow(unused, clippy::too_many_arguments)]
    fn populate_cache(
        &mut self,
        min_mz: f64,
        max_mz: f64,
        min_charge: i32,
        max_charge: i32,
        charge_carrier: f64,
        truncate_after: f64,
        ignore_below: f64,
    ) {
        tracing::trace!("No cache to populate");
    }

    /// Get the largest width isotopic pattern this generator has created so far.
    ///
    /// This is useful for optimizing how wide a window must be in order to ignore peaks
    /// but is not *necessary* to work.
    fn largest_isotopic_width(&self) -> f64 {
        f64::infinity()
    }
}

/// The mass difference between isotopes `C[13]` and `C[12]`. Not precisely universal, but the
/// majority of expected applications are carbon-based
pub const NEUTRON_SHIFT: f64 = 1.0033548378;

const ISOTOPIC_SHIFT: [f64; 10] = [
    NEUTRON_SHIFT / 1.0,
    NEUTRON_SHIFT / 2.0,
    NEUTRON_SHIFT / 3.0,
    NEUTRON_SHIFT / 4.0,
    NEUTRON_SHIFT / 5.0,
    NEUTRON_SHIFT / 6.0,
    NEUTRON_SHIFT / 7.0,
    NEUTRON_SHIFT / 8.0,
    NEUTRON_SHIFT / 9.0,
    NEUTRON_SHIFT / 10.0,
];

/// Get the m/z difference between isotopic peaks at a given charge state
#[inline(always)]
pub fn isotopic_shift(charge: i32) -> f64 {
    if charge > 0 && charge < 11 {
        ISOTOPIC_SHIFT[(charge - 1) as usize]
    } else {
        NEUTRON_SHIFT / charge as f64
    }
}

/// A package of parameters used to generate isotopic patterns
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IsotopicPatternParams {
    /// The cumulative abundance percentage of isotopic signal to retain
    pub truncate_after: f64,
    /// The minimum abundance percentage of isotopic signal a peak must have
    /// to be kept
    pub ignore_below: f64,
    /// Whether or not to use the incremental truncation algorithm, and how far
    /// down to truncate
    pub incremental_truncation: Option<f64>,
    /// The mass of the charge carrier, e.g. proton mass
    pub charge_carrier: f64,
}

impl Default for IsotopicPatternParams {
    fn default() -> Self {
        Self {
            truncate_after: 0.95,
            ignore_below: 0.001,
            incremental_truncation: None,
            charge_carrier: PROTON,
        }
    }
}

impl IsotopicPatternParams {
    pub fn new(
        truncate_after: f64,
        ignore_below: f64,
        incremental_truncation: Option<f64>,
        charge_carrier: f64,
    ) -> Self {
        Self {
            truncate_after,
            ignore_below,
            incremental_truncation,
            charge_carrier,
        }
    }
}

/// A model for converting an m/z and a theoretical charge state into a theoretical
/// isotopic pattern based upon an "average monomer" and linear extension.
///
/// This is an implementation of Senko's Averagine [^1]
///
/// # References
/// [^1]: Senko M, Beu S, McLafferty F: Determination of Monoisotopic Masses and Ion
///       Populations for Large Biomolecules from Resolved Isotopic Distributions.
///       Journal of the American Society for Mass Spectrometry 1995, 6:229-233
///       <https://doi.org/10.1016/1044-0305(95)00017-8>
#[derive(Debug, Clone)]
pub struct IsotopicModel<'lifespan> {
    /// The "average" monomer composition
    pub base_composition: FractionalComposition<'lifespan>,
    /// The mass of the average monomer to interpolate with
    pub base_mass: f64,
    hydrogen: ElementSpecification<'lifespan>,
    generator: BafflingRecursiveIsotopicPatternGenerator<'lifespan>,
}

impl PartialEq for IsotopicModel<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.base_composition == other.base_composition
    }
}

impl<'lifespan: 'transient, 'transient> IsotopicModel<'lifespan> {
    /// Create a new [`IsotopicModel`] from a fractional composition
    pub fn new<C: Into<FractionalComposition<'lifespan>>>(base_composition: C) -> Self {
        let base_composition = base_composition.into();
        Self {
            // base_mass: fractional_mass(&base_composition),
            base_mass: base_composition.mass(),
            base_composition,
            hydrogen: ElementSpecification::parse("H").unwrap(),
            generator: BafflingRecursiveIsotopicPatternGenerator::new(),
        }
    }

    pub fn scale(
        &self,
        mz: f64,
        charge: i32,
        charge_carrier: f64,
    ) -> ChemicalComposition<'transient> {
        let neutral = neutral_mass(mz, charge, charge_carrier);
        let scale = neutral / self.base_mass;

        let mut scaled = ChemicalComposition::new();
        for (elt, count) in self.base_composition.iter() {
            scaled.set(*elt, (*count * scale).round() as i32);
        }
        let scaled_mass = scaled.mass();
        let delta = (scaled_mass - neutral).round() as i32;
        let hydrogens = scaled[&self.hydrogen];
        if hydrogens > delta {
            scaled[&self.hydrogen] -= delta;
        } else {
            scaled[&self.hydrogen] = 0;
        }
        scaled
    }
}

impl IsotopicPatternGenerator for IsotopicModel<'_> {
    fn isotopic_cluster(
        &mut self,
        mz: f64,
        charge: i32,
        charge_carrier: f64,
        truncate_after: f64,
        ignore_below: f64,
    ) -> TheoreticalIsotopicPattern {
        let composition = self.scale(mz, charge, charge_carrier);
        let peaks = self
            .generator
            .isotopic_variants(composition, 0, charge, charge_carrier);
        let peaks = TheoreticalIsotopicPattern::from(peaks);
        let diff = mz - peaks.origin;
        peaks
            .truncate_after(truncate_after)
            .ignore_below(ignore_below)
            .shift(diff)
    }
}

impl<T: IntoIterator<Item = (&'static str, f64)>> From<T> for IsotopicModel<'_> {
    fn from(iter: T) -> Self {
        let mut f = FractionalComposition::default();
        for (e, c) in iter {
            f.insert(e.parse().expect("Failed to parse element specification"), c);
        }
        IsotopicModel::new(f)
    }
}

impl<'a> From<IsotopicModel<'a>> for FractionalComposition<'a> {
    fn from(value: IsotopicModel<'a>) -> Self {
        value.base_composition
    }
}

#[doc(hidden)]
// TODO: Break this into two structs, one for m/z and charge, another for the rest, and
// root the cache bucket off the latter struct, and then sort over the former struct
#[derive(Debug, Clone, Copy)]
pub struct IsotopicPatternSpec {
    pub mz: f64,
    pub charge: i32,
    pub charge_carrier: f64,
    pub truncate_after: f64,
    pub ignore_below: f64,
}

impl PartialOrd for IsotopicPatternSpec {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IsotopicPatternSpec {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.charge.cmp(&other.charge) {
            Ordering::Equal => match self.mz.total_cmp(&other.mz) {
                Ordering::Equal => match self.truncate_after.total_cmp(&other.truncate_after) {
                    Ordering::Equal => match self.ignore_below.total_cmp(&other.ignore_below) {
                        Ordering::Equal => self.charge_carrier.total_cmp(&other.charge_carrier),
                        x => x,
                    },
                    x => x,
                },
                x => x,
            },
            x => x,
        }
    }
}

impl PartialEq for IsotopicPatternSpec {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        isclose(self.mz, other.mz, 1e-6)
            && self.charge == other.charge
            && isclose(self.charge_carrier, other.charge_carrier, 1e-6)
            && isclose(self.truncate_after, other.truncate_after, 1e-6)
            && isclose(self.ignore_below, other.ignore_below, 1e-6)
    }
}

impl Eq for IsotopicPatternSpec {}

impl hash::Hash for IsotopicPatternSpec {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        let i = (self.mz * 100.0).round() as i64;
        // let i = self.mz;
        i.hash(state);
        self.charge.hash(state);
        let i = (self.truncate_after * 10.0).round() as i64;
        i.hash(state);
        let i = (self.ignore_below * 10.0).round() as i64;
        i.hash(state);
    }
}

#[cfg(feature = "experimental-partition-key")]
#[allow(unused)]
mod partition {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    pub struct IsotopicPatternCachePartitionKey {
        pub charge: i32,
        pub charge_carrier: f64,
        pub truncate_after: f64,
        pub ignore_below: f64,
    }

    impl PartialOrd for IsotopicPatternCachePartitionKey {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for IsotopicPatternCachePartitionKey {
        fn cmp(&self, other: &Self) -> Ordering {
            match self.charge.cmp(&other.charge) {
                Ordering::Equal => match self.truncate_after.total_cmp(&other.truncate_after) {
                    Ordering::Equal => match self.ignore_below.total_cmp(&other.ignore_below) {
                        Ordering::Equal => self.charge_carrier.total_cmp(&other.charge_carrier),
                        x => x,
                    },
                    x => x,
                },
                x => x,
            }
        }
    }

    impl PartialEq for IsotopicPatternCachePartitionKey {
        #[inline]
        fn eq(&self, other: &Self) -> bool {
            self.charge == other.charge
                && isclose(self.charge_carrier, other.charge_carrier, 1e-6)
                && isclose(self.truncate_after, other.truncate_after, 1e-6)
                && isclose(self.ignore_below, other.ignore_below, 1e-6)
        }
    }

    impl Eq for IsotopicPatternCachePartitionKey {}

    impl hash::Hash for IsotopicPatternCachePartitionKey {
        #[inline]
        fn hash<H: hash::Hasher>(&self, state: &mut H) {
            self.charge.hash(state);
            let i = (self.truncate_after * 10.0).round() as i64;
            i.hash(state);
            let i = (self.ignore_below * 10.0).round() as i64;
            i.hash(state);
        }
    }

    #[derive(Debug, Clone)]
    pub struct CachedIsotopicPattern {
        pub mz: f64,
        pub pattern: TheoreticalIsotopicPattern,
    }

    impl PartialEq for CachedIsotopicPattern {
        fn eq(&self, other: &Self) -> bool {
            self.mz == other.mz && self.pattern == other.pattern
        }
    }

    impl Eq for CachedIsotopicPattern {}

    impl PartialOrd for CachedIsotopicPattern {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for CachedIsotopicPattern {
        fn cmp(&self, other: &Self) -> Ordering {
            self.mz.total_cmp(&other.mz)
        }
    }

    impl PartialEq<f64> for CachedIsotopicPattern {
        fn eq(&self, other: &f64) -> bool {
            self.mz.eq(other)
        }
    }

    impl PartialOrd<f64> for CachedIsotopicPattern {
        fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
            Some(self.mz.total_cmp(other))
        }
    }

    #[derive(Debug, Clone, Default)]
    pub struct PartitionedIsotopicPatternCache {
        cache: HashMap<IsotopicPatternCachePartitionKey, Vec<CachedIsotopicPattern>>,
    }
}

#[derive(Debug, Clone, Copy)]
struct FloatRange {
    start: f64,
    end: f64,
    step: f64,
    index: usize,
}

impl FloatRange {
    fn new(start: f64, end: f64, step: f64) -> Self {
        Self {
            start,
            end,
            step,
            index: 0,
        }
    }
}

impl Iterator for FloatRange {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let val = self.start + self.step * (self.index as f64);
        if val < self.end {
            self.index += 1;
            Some(val)
        } else {
            None
        }
    }
}

/// A wrapper around [`IsotopicModel`] which includes a cache over isotopic patterns,
/// mapping similar m/z values with the same charge and parameters to a previously calculated
/// pattern if one exists, otherwise computing a new pattern and saving it in the cache.
#[derive(Debug, Clone)]
pub struct CachingIsotopicModel<'lifespan> {
    cache_truncation: f64,
    inner: IsotopicModel<'lifespan>,
    cache: BTreeMap<IsotopicPatternSpec, TheoreticalIsotopicPattern>, // See TODO for [`IsotopicPatternSpec`]
    largest_isotopic_width: Option<f64>,
}

fn isotopic_pattern_width(tid: &TheoreticalIsotopicPattern) -> f64 {
    let peaks = &tid.peaks;
    let n = peaks.len();
    if n < 2 {
        return 0.0;
    }
    unsafe {
        let first = peaks.get_unchecked(0).mz;
        let last = peaks.get_unchecked(n.saturating_sub(1)).mz;
        last - first
    }
}


impl<'lifespan> CachingIsotopicModel<'lifespan> {
    pub fn new<C: Into<FractionalComposition<'lifespan>>>(
        base_composition: C,
        cache_truncation: f64,
    ) -> Self {
        Self {
            inner: IsotopicModel::new(base_composition),
            cache: BTreeMap::new(),
            cache_truncation,
            largest_isotopic_width: Some(0.0),
        }
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn iter(&self) -> btree_map::Iter<IsotopicPatternSpec, TheoreticalIsotopicPattern> {
        self.cache.iter()
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    fn get_largest_isotopic_width(&self) -> f64 {
        self.cache
            .values()
            .map(isotopic_pattern_width)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or_default()
    }

    pub fn largest_isotopic_width(&self) -> f64 {
        self.largest_isotopic_width.unwrap_or_else(|| self.get_largest_isotopic_width())
    }

    #[inline(always)]
    fn truncate_mz(&self, mz: f64) -> f64 {
        (mz / self.cache_truncation).round() * self.cache_truncation
    }

    pub fn make_cache_key(
        &self,
        mz: f64,
        charge: i32,
        charge_carrier: f64,
        truncate_after: f64,
        ignore_below: f64,
    ) -> IsotopicPatternSpec {
        IsotopicPatternSpec {
            mz: self.truncate_mz(mz),
            charge,
            charge_carrier,
            truncate_after,
            ignore_below,
        }
    }

    pub fn populate_cache_params(
        &mut self,
        min_mz: f64,
        max_mz: f64,
        min_charge: i32,
        max_charge: i32,
        isotopic_params: IsotopicPatternParams,
    ) {
        self.populate_cache(
            min_mz,
            max_mz,
            min_charge,
            max_charge,
            isotopic_params.charge_carrier,
            isotopic_params.truncate_after,
            isotopic_params.ignore_below,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn populate_cache(
        &mut self,
        min_mz: f64,
        max_mz: f64,
        min_charge: i32,
        max_charge: i32,
        charge_carrier: f64,
        truncate_after: f64,
        ignore_below: f64,
    ) {
        let sign = min_charge / min_charge.abs();
        trace!("Starting isotopic cache population");
        for mz in FloatRange::new(min_mz, max_mz, 0.1) {
            for charge in min_charge.abs()..max_charge.abs() {
                self.isotopic_cluster(
                    mz,
                    charge * sign,
                    charge_carrier,
                    truncate_after,
                    ignore_below,
                );
            }
        }
        trace!(
            "Finished isotopic cache population, {} entries created",
            self.len()
        );
    }
}

impl IsotopicPatternGenerator for CachingIsotopicModel<'_> {
    fn largest_isotopic_width(&self) -> f64 {
        self.largest_isotopic_width()
    }

    fn isotopic_cluster(
        &mut self,
        mz: f64,
        charge: i32,
        charge_carrier: f64,
        truncate_after: f64,
        ignore_below: f64,
    ) -> TheoreticalIsotopicPattern {
        let key = self.make_cache_key(mz, charge, charge_carrier, truncate_after, ignore_below);
        match self.cache.entry(key) {
            BEntry::Occupied(ent) => {
                let res = ent.get();
                let offset = mz - res.origin;
                res.clone_shifted(offset)
            }
            BEntry::Vacant(ent) => {
                let res = self.inner.isotopic_cluster(
                    // mz,
                    key.mz,
                    charge,
                    charge_carrier,
                    truncate_after,
                    ignore_below,
                );
                let offset = mz - res.origin;
                let out = ent.insert(res).clone_shifted(offset);
                let width = isotopic_pattern_width(&out);
                match &mut self.largest_isotopic_width {
                    Some(w) => {
                        *w = w.max(width);
                    },
                    None => {
                        self.largest_isotopic_width = Some(width);
                    }
                }
                out
            }
        }
    }

    fn populate_cache(
        &mut self,
        min_mz: f64,
        max_mz: f64,
        min_charge: i32,
        max_charge: i32,
        charge_carrier: f64,
        truncate_after: f64,
        ignore_below: f64,
    ) {
        self.populate_cache(
            min_mz,
            max_mz,
            min_charge,
            max_charge,
            charge_carrier,
            truncate_after,
            ignore_below,
        )
    }
}

impl<'a> From<IsotopicModel<'a>> for CachingIsotopicModel<'a> {
    fn from(inst: IsotopicModel<'a>) -> CachingIsotopicModel<'a> {
        CachingIsotopicModel::new(inst.base_composition, 1.0)
    }
}

/// A set of named average monomer isotopic models
/// for biomolecules. Variants convert to [`IsotopicModel`]
/// and [`CachingIsotopicModel`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IsotopicModels {
    Peptide,
    Glycan,
    Glycopeptide,
    PermethylatedGlycan,
    Heparin,
    HeparanSulfate,
}

impl From<IsotopicModels> for IsotopicModel<'_> {
    fn from(source: IsotopicModels) -> IsotopicModel<'static> {
        match source {
            IsotopicModels::Peptide => vec![
                ("H", 7.7583f64),
                ("C", 4.9384),
                ("S", 0.0417),
                ("O", 1.4773),
                ("N", 1.3577),
            ],
            IsotopicModels::Glycan => vec![("H", 11.8333), ("C", 7.0), ("O", 5.16666), ("N", 0.5)],
            IsotopicModels::Glycopeptide => vec![
                ("H", 15.75),
                ("C", 10.93),
                ("S", 0.02054),
                ("O", 6.4773),
                ("N", 1.6577),
            ],
            IsotopicModels::PermethylatedGlycan => {
                vec![("C", 12.0), ("H", 21.8333), ("N", 0.5), ("O", 5.16666)]
            }
            IsotopicModels::Heparin => {
                vec![("H", 10.5), ("C", 6.0), ("S", 0.5), ("O", 5.5), ("N", 0.5)]
            }
            IsotopicModels::HeparanSulfate => vec![
                ("H", 10.667),
                ("C", 6.0),
                ("S", 1.333),
                ("O", 9.0),
                ("N", 0.667),
            ],
        }
        .into()
    }
}

impl From<IsotopicModels> for CachingIsotopicModel<'_> {
    fn from(source: IsotopicModels) -> CachingIsotopicModel<'static> {
        let model: IsotopicModel = source.into();
        model.into()
    }
}

/// Strategies for scaling a theoretical isotopic pattern to pair with an experimental
/// isotopic pattern expected to follow the same distribution.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TheoreticalIsotopicDistributionScalingMethod {
    #[default]
    /// Assume that the sum of all experimental peaks' intensities are contributed by the same isotopic pattern
    Sum,
    /// Assume the most abundant theoretical peak is the most reliable peak to determine the scaling factor for
    /// the entire theoretical distribution
    Max,
    Top3,
}

impl TheoreticalIsotopicDistributionScalingMethod {
    #[inline]
    fn find_top_3_peaks<T: IntensityMeasurement>(&self, peaks: &[T]) -> ((usize, usize, usize), (f32, f32, f32)) {
        let mut t1_index: usize = 0;
        let mut t2_index: usize = 0;
        let mut t3_index: usize = 0;
        let mut t1 = 0.0f32;
        let mut t2 = 0.0f32;
        let mut t3 = 0.0f32;

        for (i, p) in peaks.iter().enumerate() {
            let y = p.intensity();
            if y > t1 {
                t3 = t2;
                t2 = t1;
                t1 = y;
                t3_index = t2_index;
                t2_index = t1_index;
                t1_index = i;
            } else if y > t2 {
                t3_index = t2_index;
                t3 = t2;
                t2 = y;
                t2_index = i;
            } else if y > t3 {
                t3_index = i;
                t3 = y;
            }
        }

        ((t1_index, t2_index, t3_index), (t1, t2, t3))
    }

    pub fn scale<C: CentroidLike>(
        &self,
        experimental: &[C],
        theoretical: &mut TheoreticalIsotopicPattern,
    ) {
        if theoretical.is_empty() {
            return;
        }
        match self {
            Self::Sum => {
                let total: f32 = experimental.iter().map(|p| p.intensity()).sum();
                theoretical
                    .iter_mut()
                    .for_each(|p| p.intensity *= total as f64);
            }
            Self::Max => {
                let (index, theo_max) = theoretical
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.intensity().total_cmp(&b.1.intensity()))
                    .map(|(i, p)| (i, p.intensity()))
                    .unwrap();
                let scale = experimental[index].intensity() / theo_max;
                theoretical
                    .iter_mut()
                    .for_each(|p| p.intensity *= scale as f64);
            }
            Self::Top3 => {
                let ((t1_index, t2_index, t3_index), (t1, t2, t3)) = self.find_top_3_peaks(&theoretical.peaks);
                let scale = unsafe {
                    let mut scale = experimental.get_unchecked(t1_index).intensity() / t1;
                    scale += experimental.get_unchecked(t2_index).intensity() / t2;
                    scale += experimental.get_unchecked(t3_index).intensity() / t3;
                    scale / 3.0
                };
                eprintln!("Scale term: {scale}");
                theoretical
                    .iter_mut()
                    .for_each(|p| p.intensity *= scale as f64);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BasePeakToMonoisotopicOffsetEstimator<I: IsotopicPatternGenerator> {
    isotopic_model: I,
    bins: Vec<usize>,
    step_size: f64,
}

impl<I: IsotopicPatternGenerator> BasePeakToMonoisotopicOffsetEstimator<I> {
    pub fn new(isotopic_model: I, step_size: f64) -> Self {
        Self {
            isotopic_model,
            bins: Vec::new(),
            step_size,
        }
    }

    pub fn len(&self) -> usize {
        self.bins.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bins.is_empty()
    }

    fn max_bin_mass(&self) -> f64 {
        self.len() as f64 * self.step_size
    }

    fn bin_for(&self, mass: f64) -> usize {
        let offset = mass / self.step_size;
        offset as usize
    }

    fn estimate_for_peak_offset(&mut self, mass: f64) -> usize {
        let tid = self
            .isotopic_model
            .isotopic_cluster(mass, 1, PROTON, 0.99, 0.0);
        tid.into_iter()
            .enumerate()
            .max_by(|(_, pa), (_, pb)| pa.intensity.total_cmp(&pb.intensity))
            .map(|(i, _)| i)
            .unwrap()
    }

    fn populate_bins(&mut self, max_mass: f64) {
        let mut current_bin = self.max_bin_mass();

        while max_mass >= current_bin {
            let next_bin_mass = current_bin + self.step_size;
            let delta = self.estimate_for_peak_offset(next_bin_mass);
            self.bins.push(delta);
            current_bin = next_bin_mass;
        }
    }

    pub fn get_peak_offset(&mut self, mass: f64) -> usize {
        let index = self.bin_for(mass);
        if self.len() <= index {
            self.populate_bins(mass);
        };
        self.bins[index]
    }

    pub fn get_peak_offset_direct(&mut self, mass: f64) -> usize {
        self.estimate_for_peak_offset(mass)
    }
}

#[cfg(test)]
mod test {
    use mzpeaks::{CentroidPeak, MZPeakSetType};

    use super::*;
    use std::hash::{Hash, Hasher};

    macro_rules! assert_is_close {
        ($t1:expr, $t2:expr, $tol:expr, $label:literal) => {
            assert!(
                ($t1 - $t2).abs() < $tol,
                "Observed {} {}, expected {}, difference {}",
                $label,
                $t1,
                $t2,
                $t1 - $t2,
            );
        };
    }

    #[test]
    fn test_fc() {
        let model: IsotopicModel = IsotopicModels::Peptide.into();
        assert!(model.base_composition.contains_key("C"));
        assert_eq!(model.base_composition.get("C").copied(), Some(4.9384));
        assert_eq!(model.base_composition.len(), 5);
    }

    #[test]
    fn test_tid() {
        let mut model: IsotopicModel = IsotopicModels::Peptide.into();
        let tid = model
            .isotopic_cluster(1000.0, 2, PROTON, 0.95, 0.001)
            .scale_by(100.0);

        assert!((tid.total() - 100.0).abs() < 1e-6);
        assert!((tid[0].mz() - 1000.0).abs() < 1e-6);

        let diff = tid[1].mz() - tid[0].mz();
        let neutron2 = 0.5014313;

        assert!((diff - neutron2).abs() < 1e-6);
        let expected = 31.296387;
        assert!(
            (tid[0].intensity() - expected).abs() < 1e-6,
            "{} - {expected} = {}",
            tid[0].intensity(),
            tid[0].intensity() - expected
        );

        let tid = model
            .isotopic_cluster(1000.5, 2, PROTON, 0.95, 0.001)
            .scale_by(100.0);

        assert!((tid.total() - 100.0).abs() < 1e-6);
        assert!((tid[0].mz() - 1000.5).abs() < 1e-6);

        let diff = tid[1].mz() - tid[0].mz();
        let neutron2 = 0.5014313;

        assert!((diff - neutron2).abs() < 1e-6);
        let expected = 31.29292;
        assert!(
            (tid[0].intensity() - expected).abs() < 1e-6,
            "{} - {expected} = {}",
            tid[0].intensity(),
            tid[0].intensity() - expected
        );
    }

    #[test]
    fn test_scale() {
        let mut isomod: IsotopicModel = IsotopicModels::Heparin.into();
        let mut ref_tid = isomod.isotopic_cluster(1500.0, 2, PROTON, 0.95, 0.1);
        ref_tid = ref_tid.scale_by(1000.0);
        let eid: MZPeakSetType<_> = ref_tid.iter().map(|p| CentroidPeak::new(p.mz(), p.intensity(), 0)).collect();

        let mut tid = isomod.isotopic_cluster(1500.0, 2, PROTON, 0.95, 0.1);
        TheoreticalIsotopicDistributionScalingMethod::Sum.scale(eid.as_slice(), &mut tid);

        let tic_ref: f32 = ref_tid.iter().map(|p| p.intensity()).sum();
        let tic_sum: f32 = tid.iter().map(|p| p.intensity()).sum();
        assert!((tic_ref - tic_sum).abs() < 1e-3);

        let mut tid = isomod.isotopic_cluster(1500.0, 2, PROTON, 0.95, 0.1);
        TheoreticalIsotopicDistributionScalingMethod::Max.scale(eid.as_slice(), &mut tid);

        let tic_max: f32 = tid.iter().map(|p| p.intensity()).sum();
        assert!((tic_ref - tic_max).abs() < 1e-3);

        let mut tid = isomod.isotopic_cluster(1500.0, 2, PROTON, 0.95, 0.1);
        TheoreticalIsotopicDistributionScalingMethod::Top3.scale(eid.as_slice(), &mut tid);

        let tic_top3: f32 = tid.iter().map(|p| p.intensity()).sum();
        assert!((tic_ref - tic_top3).abs() < 1e-3, "Observed TIC {tic_top3}, expected TIC {tic_ref}, delta: {}", tic_ref - tic_top3);
    }

    #[test]
    fn test_cache_key() {
        let model: CachingIsotopicModel = IsotopicModels::Peptide.into();
        let key1 = model.make_cache_key(1000.0, 2, PROTON, 0.95, 0.001);
        let key2 = model.make_cache_key(1000.0, 2, PROTON, 0.95, 0.001);
        assert_eq!(key1, key2);

        let mut hasher1 = hash_map::DefaultHasher::default();
        key1.hash(&mut hasher1);

        let mut hasher2 = hash_map::DefaultHasher::default();
        key2.hash(&mut hasher2);

        let v1 = hasher1.finish();
        let v2 = hasher2.finish();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_conversion_for_frac_composition() {
        let parts = [
            ("H", 10.667),
            ("C", 6.0),
            ("S", 1.333),
            ("O", 9.0),
            ("N", 0.667),
        ];
        let parts = parts.map(|(e, c)| {
            (e.parse::<ElementSpecification>().unwrap(), c)
        });
        let hash_map: HashMap<_, _> = parts.into_iter().collect();
        let frac_comp: FractionalComposition = hash_map.into();
        let hs_isomod: IsotopicModel = IsotopicModels::HeparanSulfate.into();
        assert_eq!(hs_isomod.base_composition, frac_comp);

        let frac_comp: FractionalComposition = parts.into_iter().collect();
        assert_eq!(hs_isomod.base_composition, frac_comp);

        let frac_comp: FractionalComposition = parts.to_vec().into();
        assert_eq!(hs_isomod.base_composition, frac_comp);
    }

    #[test]
    fn test_mz_truncation() {
        let mut model: CachingIsotopicModel = IsotopicModels::Peptide.into();
        model.cache_truncation = 1.0;
        let x = model.truncate_mz(1000.0);
        assert_eq!(x, 1000.0);
        let x = model.truncate_mz(1000.1);
        assert_eq!(x, 1000.0);
    }

    #[test]
    fn test_cache() {
        let mut model: CachingIsotopicModel = IsotopicModels::Peptide.into();
        assert_eq!(model.get_largest_isotopic_width(), 0.0);
        assert!(model.is_empty());

        model.populate_cache(200.0, 400.0, 2, 4, PROTON, 0.95, 0.001);
        assert_is_close!(model.get_largest_isotopic_width(), 1.0027695470485014, 1e-6, "width");

        model.clear();
        assert_eq!(model.get_largest_isotopic_width(), 0.0);
        assert!(model.is_empty());

        let mut model: IsotopicModel = IsotopicModels::Peptide.into();
        assert_eq!(model.largest_isotopic_width(), f64::INFINITY);
        model.populate_cache(200.0, 400.0, 2, 4, PROTON, 0.95, 0.001);
        assert_eq!(model.largest_isotopic_width(), f64::INFINITY);
    }

    #[test]
    fn test_tid_cached() {
        let mut model: CachingIsotopicModel = IsotopicModels::Peptide.into();
        let tid = model
            .isotopic_cluster(1000.0, 2, PROTON, 0.95, 0.001)
            .scale_by(100.0);

        assert!((tid.total() - 100.0).abs() < 1e-6);
        assert!((tid[0].mz() - 1000.0).abs() < 1e-6);

        let tot: f32 = tid.iter().map(|p| p.intensity()).sum();
        assert!((tot - 100.0).abs() < 1e-3);

        let diff = tid[1].mz() - tid[0].mz();
        let neutron2 = 0.5014313;

        assert!((diff - neutron2).abs() < 1e-6);
        let expected = 31.296387;
        assert!(
            (tid[0].intensity() - expected).abs() < 1e-6,
            "{} - {expected} = {}",
            tid[0].intensity(),
            tid[0].intensity() - expected
        );

        let tid = model
            .isotopic_cluster(1000.001, 2, PROTON, 0.95, 0.001)
            .scale_by(100.0);

        assert!((tid.total() - 100.0).abs() < 1e-6);
        assert!((tid[0].mz() - 1000.001).abs() < 1e-6);

        let tot: f32 = tid.iter().map(|p| p.intensity()).sum();
        assert!((tot - 100.0).abs() < 1e-3);

        let diff = tid[1].mz() - tid[0].mz();
        let neutron2 = 0.5014313;

        assert!((diff - neutron2).abs() < 1e-6);
        let expected = 31.296387;
        assert!(
            (tid[0].intensity() - expected).abs() < 1e-6,
            "{} - {expected} = {}",
            tid[0].intensity(),
            tid[0].intensity() - expected
        );
        assert_eq!(model.len(), 1);

        let tid = model.isotopic_cluster(1000.0 - 0.001, 2, PROTON, 0.95, 0.001);
        assert_eq!(model.len(), 1);
        let tid = tid.scale_by(100.0);
        let tot: f32 = tid.iter().map(|p| p.intensity()).sum();
        assert!((tot - 100.0).abs() < 1e-3);
    }

    #[test]
    fn test_mono_to_base() {
        let mut model: BasePeakToMonoisotopicOffsetEstimator<IsotopicModel> =
            BasePeakToMonoisotopicOffsetEstimator::new(IsotopicModels::Peptide.into(), 10.0);

        assert_eq!(model.get_peak_offset(1000.0), 0);
        assert_eq!(model.get_peak_offset(2500.0), 1);
        assert_eq!(model.get_peak_offset(10000.0), 6);
    }
}
