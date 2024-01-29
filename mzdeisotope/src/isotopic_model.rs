/*! Isotopic models for generating isotopic patterns */
use std::cmp::Ordering;
use std::collections::btree_map::{self, BTreeMap, Entry as BEntry};
#[allow(unused)]
use std::collections::hash_map::{self, Entry, HashMap};
use std::hash;

use chemical_elements::isotopic_pattern::{
    BafflingRecursiveIsotopicPatternGenerator, TheoreticalIsotopicPattern,
};
use chemical_elements::{
    neutral_mass, ChemicalComposition, ElementSpecification, PROTON as _PROTON,
};

use mzpeaks::CentroidLike;
use num_traits::Float;

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
    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&f64>
    where
        ElementSpecification<'a>: std::borrow::Borrow<Q>,
        Q: hash::Hash + Eq,
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

    pub fn contains_key<Q: ?Sized>(&self, k: &Q) -> bool
    where
        ElementSpecification<'a>: std::borrow::Borrow<Q>,
        Q: hash::Hash + Eq,
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
    #[allow(unused)]
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
        tracing::warn!("No cache to populate");
    }
}


/// The mass difference between isotopes C[13] and C[12]. Not precisely universal, but the
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
/// This is an implementation of [Senko's Averagine][1]
///
/// # References
/// - [1]: <https://doi.org/10.1016/1044-0305(95)00017-8>
///     Senko M, Beu S, McLafferty F: Determination of Monoisotopic Masses and Ion
///     Populations for Large Biomolecules from Resolved Isotopic Distributions.
///     Journal of the American Society for Mass Spectrometry 1995, 6:229-233
#[derive(Debug, Clone)]
pub struct IsotopicModel<'lifespan> {
    /// The "average" monomer composition
    pub base_composition: FractionalComposition<'lifespan>,
    /// The mass of the average monomer to interpolate with
    pub base_mass: f64,
    hydrogen: ElementSpecification<'lifespan>,
    generator: BafflingRecursiveIsotopicPatternGenerator<'lifespan>,
}

impl<'lifespan> PartialEq for IsotopicModel<'lifespan> {
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

impl<'lifespan> IsotopicPatternGenerator for IsotopicModel<'lifespan> {
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

impl<'lifespan, T: IntoIterator<Item = (&'static str, f64)>> From<T> for IsotopicModel<'lifespan> {
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
}

impl<'lifespan: 'transient, 'transient> CachingIsotopicModel<'lifespan> {
    pub fn new<C: Into<FractionalComposition<'lifespan>>>(
        base_composition: C,
        cache_truncation: f64,
    ) -> Self {
        Self {
            inner: IsotopicModel::new(base_composition),
            cache: BTreeMap::new(),
            cache_truncation,
        }
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn iter(&self) -> btree_map::Iter<IsotopicPatternSpec, TheoreticalIsotopicPattern> {
        self.cache.iter()
    }

    pub fn clear(&mut self) {
        self.cache.clear();
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
        tracing::trace!("Starting isotopic cache population");
        FloatRange::new(min_mz, max_mz, 0.1)
            .into_iter()
            .for_each(|mz| {
                (min_charge.abs()..max_charge.abs()).for_each(|charge| {
                    self.isotopic_cluster(
                        mz,
                        charge * sign,
                        charge_carrier,
                        truncate_after,
                        ignore_below,
                    );
                });
            });
        tracing::trace!(
            "Finished isotopic cache population, {} entries created",
            self.len()
        );
    }
}

impl<'lifespan> IsotopicPatternGenerator for CachingIsotopicModel<'lifespan> {
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

#[derive(Debug, Clone, Copy)]
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
    pub fn scale<C: CentroidLike>(
        &self,
        experimental: &[C],
        theoretical: &mut TheoreticalIsotopicPattern,
    ) {
        if theoretical.len() == 0 {
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
                    .and_then(|(i, p)| Some((i, p.intensity())))
                    .unwrap();
                // let (index, peak) = experimental
                //     .iter()
                //     .enumerate()
                //     .max_by(|a, b| a.1.intensity().partial_cmp(&b.1.intensity()).unwrap())
                //     .unwrap();
                // let scale = peak.intensity() / experimental[index].intensity();
                let scale = experimental[index].intensity() / theo_max;
                theoretical
                    .iter_mut()
                    .for_each(|p| p.intensity *= scale as f64);
            }
            Self::Top3 => {
                let mut t1_index: usize = 0;
                let mut t2_index: usize = 0;
                let mut t3_index: usize = 0;
                let mut t1 = 0.0f32;
                let mut t2 = 0.0f32;
                let mut t3 = 0.0f32;

                for (i, p) in experimental.iter().enumerate() {
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

                let mut scale = experimental[t1_index].intensity() / t1;
                scale += experimental[t2_index].intensity() / t2;
                scale += experimental[t3_index].intensity() / t3;
                theoretical
                    .iter_mut()
                    .for_each(|p| p.intensity *= scale as f64);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::hash::{Hash, Hasher};

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
    fn test_mz_truncation() {
        let mut model: CachingIsotopicModel = IsotopicModels::Peptide.into();
        model.cache_truncation = 1.0;
        let x = model.truncate_mz(1000.0);
        assert_eq!(x, 1000.0);
        let x = model.truncate_mz(1000.1);
        assert_eq!(x, 1000.0);
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
}
