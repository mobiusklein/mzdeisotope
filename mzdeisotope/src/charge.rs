/*! Utilities for describing charge state ranges and calculations */
use std::cmp;

use thiserror::Error;

use mzpeaks::prelude::*;

/// A charge range is just a pair of integers
pub type ChargeRange = (i32, i32);

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum ChargeRangeError {
    #[error("Charge range cannot span zero")]
    ChargeCannotBeZero,
    #[error("Both min and max charge state must have the same sign")]
    SignsMustMatch,
}

/// A marker trait indicating that an iterator can be used to produce charge states
pub trait ChargeIterator: Iterator<Item = i32> {}

/// An iterator for a series of contiguous charge states within a range
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

#[cfg(feature = "charge-v2")]
mod v2 {
    use super::*;


#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChargeRangeV2(pub i32, pub i32);

impl From<ChargeRangeV2> for ChargeRange {
    fn from(value: ChargeRangeV2) -> Self {
        (value.0, value.1)
    }
}


impl TryFrom<(i32, i32)> for ChargeRangeV2 {
    type Error = ChargeRangeError;

    fn try_from(value: (i32, i32)) -> Result<Self, Self::Error> {
        let (a, b) = value;
        if a.signum() != b.signum() {
            Err(ChargeRangeError::SignsMustMatch)
        } else if a == 0 || b == 0 {
            Err(ChargeRangeError::ChargeCannotBeZero)
        } else if a.abs() < b.abs() {
            Ok(ChargeRangeV2(a, b))
        } else {
            Ok(ChargeRangeV2(b, a))
        }
    }
}

impl ChargeRangeV2 {
    pub fn new(low: i32, high: i32) -> Result<Self, ChargeRangeError> {
        (low, high).try_into()
    }

    pub fn iter(&self) -> ChargeRangeIter {
        ChargeRangeIter::new(self.0, self.1)
    }

    pub fn abs(&self) -> ChargeRangeV2 {
        Self::new(self.0.abs(), self.1.abs()).unwrap()
    }
}

impl IntoIterator for ChargeRangeV2 {
    type Item = i32;

    type IntoIter = ChargeRangeIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for &ChargeRangeV2 {
    type Item = i32;

    type IntoIter = ChargeRangeIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl From<ChargeRangeV2> for ChargeRangeIter {
    fn from(value: ChargeRangeV2) -> Self {
        value.into_iter()
    }
}

}

/// Hoopman's [QuickCharge][1] algorithm to quickly estimate the charge states to actually try
/// fitting a given seed peak with.
///
/// For safety, this does not rule out charge state 1 unless 1 is not in `charge_range`.
///
/// # Arguments
/// - `peaks`: The centroid mass spectrum being deconvoluted
/// - `position`: The index of the peak to be checked
/// - `charge_range`: The range of charge states to actually consider
///
/// ## Static Parameter
/// - `N`: The maximum charge state to allocate a solution for. Rust's const expressions
///        are not yet good enough to let us do more with this beyond a small memory
///        optimization
///
/// # Caveats
/// [`mzdeisotope`](crate) fits relative to the monoisotopic peak, and if A+1
/// peak is missing due to interference or some other signal cleaning failure
/// then the *true* charge state will be missing.
///
/// # References
/// - [1]: <https://doi.org/10.1021/ac0700833>
///        Hoopmann, M. R., Finney, G. L., MacCoss, M. J., Michael R. Hoopmann, Gregory L. Finney,
///        and, MacCoss*, M. J., … MacCoss, M. J. (2007). "High-speed data reduction, feature detection
///        and MS/MS spectrum quality assessment of shotgun proteomics data sets using high-resolution
///        Mass Spectrometry". Analytical Chemistry, 79(15), 5620–5632. <https://doi.org/10.1021/ac0700833>
///
pub fn quick_charge<C: CentroidLike, const N: usize>(
    peaks: &[C],
    position: usize,
    charge_range: ChargeRange,
) -> ChargeListIter {
    let (min_charge, max_charge) = charge_range;
    let mut charges = [false; N];
    if N > 0 {
        charges[0] = true;
    }
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
        if !charges[(charge - 1) as usize] {
            result_size += 1;
        }
        charges[(charge - 1) as usize] = true;
    }

    let mut result = Vec::with_capacity(result_size);
    charges.iter().enumerate().for_each(|(j, hit)| {
        let z = (j + 1) as i32;
        if *hit && accept_charge(z, &charge_range) {
            result.push(z)
        }
    });
    result.into()
}

#[inline(always)]
fn accept_charge(z: i32, charge_range: &ChargeRange) -> bool {
    let z = z.abs();
    charge_range.0.abs() <= z && z <= charge_range.1.abs()
}

/// A [`ChargeIterator`] implementation for a sequence of charge values that are not necessarily
/// contiguous.
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

impl ChargeIterator for std::vec::IntoIter<i32> {}

/// An wrapper around [`quick_charge`] which dispatches to an appropriate staticly compiled
/// variant with minimal stack allocation.
pub fn quick_charge_w<C: CentroidLike>(
    peaks: &[C],
    position: usize,
    charge_range: ChargeRange,
) -> ChargeListIter {
    macro_rules! match_i {
        ($($i:literal, )*) => {
            match charge_range.1 {
                $($i => quick_charge::<C, $i>(peaks, position, charge_range),)*
                i if i > 16 && i < 33 => quick_charge::<C, 32>(peaks, position, charge_range),
                i if i > 32 && i < 65 => quick_charge::<C, 64>(peaks, position, charge_range),
                i if i > 64 && i < 129 => quick_charge::<C, 128>(peaks, position, charge_range),
                _ => quick_charge::<C, 256>(peaks, position, charge_range),
            }
        };
    }
    match_i!(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,)
}

#[cfg(test)]
mod test {
    use std::fs;
    use std::io;
    use std::io::BufRead;

    use flate2::bufread::GzDecoder;

    use mzdata::MzMLReader;

    use super::*;

    #[test]
    fn test_quick_charge() -> io::Result<()> {
        let decoder = GzDecoder::new(io::BufReader::new(fs::File::open(
            "./tests/data/20150710_3um_AGP_001_29_30.mzML.gz",
        )?));

        let mut fh = io::BufReader::new(fs::File::open("./tests/data/charges.txt")?);
        let mut line = String::new();
        let mut reader = MzMLReader::new(decoder);
        let scan = reader.next().unwrap();
        let centroided = scan.into_centroid().unwrap();

        let peaks = &centroided.peaks[0..];

        for i in 0..peaks.len() {
            let charge_list: Vec<_> = quick_charge_w(peaks, i, (1, 8)).collect();
            line.clear();
            fh.read_line(&mut line)?;
            line = line.replace('\r', "");
            if line == "\n" {
                assert_eq!(charge_list, vec![1]);
            } else {
                let mut expected: Vec<_> = line
                    .strip_suffix("\n")
                    .unwrap()
                    .split(',')
                    .map(|t| {
                        t.parse::<i32>().unwrap_or_else(|e| {
                            panic!("Failed to parse line {} ({}): {}", i, line, e)
                        })
                    })
                    .collect();
                if *expected.first().unwrap() != 1 {
                    let mut tmp = vec![1];
                    tmp.extend(expected);
                    expected = tmp;
                }
                assert_eq!(charge_list, expected);
            }
        }

        Ok(())
    }
}
