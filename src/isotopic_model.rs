use chemical_elements::isotopic_pattern::{
    BafflingRecursiveIsotopicPatternGenerator, TheoreticalIsotopicPattern,
};
use chemical_elements::{
    neutral_mass, ChemicalComposition, ElementSpecification, PROTON as _PROTON,
};
use std::collections::hash_map::{Entry, HashMap};

pub const PROTON: f64 = _PROTON;

pub type FractionalComposition<'a> = HashMap<ElementSpecification<'a>, f64>;

fn fractional_mass(comp: &FractionalComposition) -> f64 {
    comp.iter()
        .map(|(e, c)| e.element.most_abundant_mass * *c)
        .sum()
}

pub trait IsotopicPatternGenerator {
    fn isotopic_cluster(
        &mut self,
        mz: f64,
        charge: i32,
        charge_carrier: f64,
        truncate_after: f64,
        ignore_below: f64,
    ) -> TheoreticalIsotopicPattern;
}

pub const NEUTRON_SHIFT: f64 = 1.0033548378;

pub fn isotopic_shift(charge: i32) -> f64 {
    NEUTRON_SHIFT / charge as f64
}

#[derive(Debug, Clone)]
pub struct IsotopicModel<'lifespan> {
    pub base_composition: FractionalComposition<'lifespan>,
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
    pub fn new(base_composition: FractionalComposition<'lifespan>) -> Self {
        Self {
            base_mass: fractional_mass(&base_composition),
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
            scaled.set(elt.clone(), (*count as f64 * scale).round() as i32);
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
        let mut f = FractionalComposition::new();
        for (e, c) in iter {
            f.insert(e.parse().expect("Failed to parse element specification"), c);
        }
        IsotopicModel::new(f)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IsotopicPatternSpec {
    pub mz: f64,
    pub charge: i32,
    pub charge_carrier: f64,
    pub truncate_after: f64,
    pub ignore_below: f64,
}

impl PartialEq for IsotopicPatternSpec {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if (self.mz - other.mz).abs() > 1e-6 {
            false
        } else if self.charge != other.charge {
            false
        } else if (self.charge_carrier - other.charge_carrier).abs() > 1e-6 {
            false
        } else if (self.truncate_after - other.truncate_after).abs() > 1e-6 {
            false
        } else if (self.ignore_below - other.ignore_below).abs() > 1e-6 {
            false
        } else {
            true
        }
    }
}

impl Eq for IsotopicPatternSpec {}

impl std::hash::Hash for IsotopicPatternSpec {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let i = (self.mz * 100.0).round() as i64;
        i.hash(state);
        self.charge.hash(state);
        let i = (self.truncate_after * 10.0).round() as i64;
        i.hash(state);
    }
}

#[derive(Debug, Clone)]
pub struct CachingIsotopicModel<'lifespan> {
    pub cache_truncation: f64,
    inner: IsotopicModel<'lifespan>,
    cache: HashMap<IsotopicPatternSpec, TheoreticalIsotopicPattern>,
}

impl<'lifespan: 'transient, 'transient> CachingIsotopicModel<'lifespan> {
    pub fn new(base_composition: FractionalComposition<'lifespan>, cache_truncation: f64) -> Self {
        Self {
            inner: IsotopicModel::new(base_composition),
            cache: HashMap::new(),
            cache_truncation: cache_truncation,
        }
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    pub fn populate(
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
        for mz in (min_mz as i64)..(max_mz as i64) {
            for charge in min_charge.abs()..max_charge.abs() {
                self.isotopic_cluster(
                    mz as f64,
                    charge * sign,
                    charge_carrier,
                    truncate_after,
                    ignore_below,
                );
            }
        }
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
        let key = IsotopicPatternSpec {
            mz: (mz * self.cache_truncation).round() / self.cache_truncation,
            charge,
            charge_carrier,
            truncate_after,
            ignore_below,
        };
        match self.cache.entry(key) {
            Entry::Occupied(ent) => {
                let res = ent.get();
                let offset = mz - res.origin;
                return res.clone_shifted(offset);
            }
            Entry::Vacant(ent) => {
                let res = self.inner.isotopic_cluster(
                    mz,
                    charge,
                    charge_carrier,
                    truncate_after,
                    ignore_below,
                );
                let offset = mz - res.origin;
                let out = ent.insert(res).clone_shifted(offset);
                return out;
            }
        }
    }
}

impl<'a> From<IsotopicModel<'a>> for CachingIsotopicModel<'a> {
    fn from(inst: IsotopicModel<'a>) -> CachingIsotopicModel<'a> {
        CachingIsotopicModel::new(inst.base_composition, 10.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum IsotopicModels {
    Peptide,
    Glycan,
    Glycopeptide,
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

#[cfg(test)]
mod test {
    use super::*;

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
        assert!((tid[0].intensity() - 32.476418).abs() < 1e-6);

        let tid = model
            .isotopic_cluster(1000.5, 2, PROTON, 0.95, 0.001)
            .scale_by(100.0);

        assert!((tid.total() - 100.0).abs() < 1e-6);
        assert!((tid[0].mz() - 1000.5).abs() < 1e-6);

        let diff = tid[1].mz() - tid[0].mz();
        let neutron2 = 0.5014313;

        assert!((diff - neutron2).abs() < 1e-6);
        assert!((tid[0].intensity() - 32.47306).abs() < 1e-6);
    }

    #[test]
    fn test_tid_cached() {
        let mut model: CachingIsotopicModel = IsotopicModels::Peptide.into();
        let tid = model
            .isotopic_cluster(1000.0, 2, PROTON, 0.95, 0.001)
            .scale_by(100.0);

        assert!((tid.total() - 100.0).abs() < 1e-6);
        assert!((tid[0].mz() - 1000.0).abs() < 1e-6);

        let diff = tid[1].mz() - tid[0].mz();
        let neutron2 = 0.5014313;

        assert!((diff - neutron2).abs() < 1e-6);
        assert!((tid[0].intensity() - 32.476418).abs() < 1e-6);

        let tid = model
            .isotopic_cluster(1000.001, 2, PROTON, 0.95, 0.001)
            .scale_by(100.0);

        assert!((tid.total() - 100.0).abs() < 1e-6);
        assert!((tid[0].mz() - 1000.001).abs() < 1e-6);

        let diff = tid[1].mz() - tid[0].mz();
        let neutron2 = 0.5014313;

        assert!((diff - neutron2).abs() < 1e-6);
        assert!((tid[0].intensity() - 32.476418).abs() < 1e-6);
    }
}
