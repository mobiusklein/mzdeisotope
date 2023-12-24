use mzpeaks::prelude::*;
use std::cmp;

pub type ChargeRange = (i32, i32);


pub trait ChargeIterator: Iterator<Item = i32> {}

#[derive(Debug, Clone)]
pub struct ChargeRangeIter {
    pub min: i32,
    pub max: i32,
    pub sign: i32,
    index: usize,
    size: usize,
}

impl ChargeRangeIter {
    pub fn new(min: i32, max: i32) -> ChargeRangeIter {
        let low = cmp::min(min.abs(), max.abs());
        let high = cmp::max(min.abs(), max.abs());
        let sign = min / min.abs();
        let size = (high - low) as usize;
        ChargeRangeIter {
            min: low,
            max: high,
            sign,
            index: 0,
            size,
        }
    }

    pub fn next_charge(&mut self) -> Option<i32> {
        if self.index >= self.size {
            None
        } else {
            let i = (self.min + self.index as i32) * self.sign;
            self.index += 1;
            Some(i)
        }
    }
}

impl Iterator for ChargeRangeIter {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        self.next_charge()
    }
}

impl From<ChargeRange> for ChargeRangeIter {
    fn from(pair: ChargeRange) -> ChargeRangeIter {
        ChargeRangeIter::new(pair.0, pair.1)
    }
}

impl ChargeIterator for ChargeRangeIter {}

pub fn quick_charge<C: CentroidLike, const N: usize>(
    peaks: &[C],
    position: usize,
    charge_range: ChargeRange
) -> ChargeListIter {
    let (min_charge, max_charge) = charge_range;
    let mut charges = [false; N];
    let peak = &peaks[position];
    let mut result_size = 0usize;
    let min_intensity = peak.intensity() / 4.0;
    for other in peaks.iter().skip(position + 1) {
        if other.intensity() < min_intensity {
            continue;
        }
        let diff = other.mz() - peak.mz();
        if diff > 1.1 {
            break;
        }
        let raw_charge = 1.0 / diff;
        let charge = (raw_charge + 0.5) as i32;
        let remain = raw_charge - raw_charge.floor();
        if 0.2 < remain && remain < 0.8 {
            continue;
        }
        if charge < min_charge || charge > max_charge {
            continue;
        }
        if !charges[charge as usize] {
            result_size += 1;
        }
        charges[charge as usize] = true;
    }

    let mut result = Vec::with_capacity(result_size);
    charges.iter().enumerate().for_each(|(j, hit)| {
        if *hit {
            result.push(j as i32)
        }
    });
    result.into()
}

#[derive(Debug, Clone)]
pub struct ChargeListIter {
    valid: Vec<i32>,
    index: usize,
}

impl ChargeListIter {
    pub fn new(valid: Vec<i32>) -> Self {
        Self { valid, index: 0 }
    }

    pub fn next_charge(&mut self) -> Option<i32> {
        if self.index < self.valid.len() {
            let val = self.valid[self.index];
            self.index += 1;
            Some(val)
        } else {
            None
        }
    }
}

impl Iterator for ChargeListIter {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        self.next_charge()
    }
}

impl ChargeIterator for ChargeListIter {}

impl From<Vec<i32>> for ChargeListIter {
    fn from(value: Vec<i32>) -> Self {
        Self::new(value)
    }
}


#[derive(Debug, Clone, Copy, Default)]
pub enum ChargeStrategy {
    #[default]
    ChargeRange,
    QuickCharge
}

impl ChargeStrategy {
    pub fn for_peak<C: CentroidLike>(&self, peaks: &[C], position: usize, charge_range: ChargeRange) -> Box<dyn ChargeIterator> {
        match self {
            ChargeStrategy::ChargeRange => Box::from(ChargeRangeIter::new(charge_range.0, charge_range.1)),
            ChargeStrategy::QuickCharge => Box::from(quick_charge::<C, 128>(peaks, position, charge_range)),
        }
    }
}