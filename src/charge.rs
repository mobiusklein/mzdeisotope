use mzpeaks::prelude::*;
use mzpeaks::MZPeakSetType;
use std::cmp;
use std::ops::Index;

pub trait ChargeIterator: Iterator<Item = i32> {}

#[derive(Debug, Clone)]
pub struct ChargeRangeIter {
    pub min: i32,
    pub max: i32,
    pub sign: i8,
    index: usize,
    size: usize,
}

impl ChargeRangeIter {
    pub fn new(min: i32, max: i32) -> ChargeRangeIter {
        let low = cmp::min(min.abs(), max.abs());
        let high = cmp::max(min.abs(), max.abs());
        let sign = (min / min.abs()) as i8;
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
            let i = (self.min + self.index as i32) * self.sign as i32;
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

impl From<(i32, i32)> for ChargeRangeIter {
    fn from(pair: (i32, i32)) -> ChargeRangeIter {
        ChargeRangeIter::new(pair.0, pair.1)
    }
}

impl ChargeIterator for ChargeRangeIter {}

pub fn quick_charge<C: CentroidLike, const N: usize>(
    peaks: &MZPeakSetType<C>,
    position: usize,
    min_charge: i32,
    max_charge: i32,
) -> Vec<i32> {
    let mut charges = [false; N];
    let peak = &peaks[position];
    let mut result_size = 0usize;
    let min_intensity = peak.intensity() / 4.0;
    for j in (position + 1)..peaks.len() {
        let other = peaks.index(j);
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
    for j in 0..N {
        if charges[j] {
            result.push(j as i32)
        }
    }
    result
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
