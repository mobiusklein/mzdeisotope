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
    charge_range: ChargeRange,
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
        if !charges[(charge - 1) as usize] {
            result_size += 1;
        }
        charges[(charge - 1) as usize] = true;
    }

    let mut result = Vec::with_capacity(result_size);
    charges.iter().enumerate().for_each(|(j, hit)| {
        if *hit {
            result.push((j + 1) as i32)
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

pub fn quick_charge_w<C: CentroidLike>(
    peaks: &[C],
    position: usize,
    charge_range: ChargeRange,
) -> ChargeListIter {
    match charge_range.1 {
        4 => quick_charge::<C, 4>(peaks, position, charge_range),
        5 => quick_charge::<C, 5>(peaks, position, charge_range),
        6 => quick_charge::<C, 6>(peaks, position, charge_range),
        7 => quick_charge::<C, 7>(peaks, position, charge_range),
        8 => quick_charge::<C, 8>(peaks, position, charge_range),
        9 => quick_charge::<C, 9>(peaks, position, charge_range),
        10 => quick_charge::<C, 10>(peaks, position, charge_range),
        11 => quick_charge::<C, 11>(peaks, position, charge_range),
        12 => quick_charge::<C, 12>(peaks, position, charge_range),
        13 => quick_charge::<C, 13>(peaks, position, charge_range),
        14 => quick_charge::<C, 14>(peaks, position, charge_range),
        15 => quick_charge::<C, 15>(peaks, position, charge_range),
        16 => quick_charge::<C, 16>(peaks, position, charge_range),
        _ => quick_charge::<C, 128>(peaks, position, charge_range),
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub enum ChargeStrategy {
    #[default]
    ChargeRange,
    QuickCharge,
}

impl ChargeStrategy {
    pub fn for_peak<C: CentroidLike>(
        &self,
        peaks: &[C],
        position: usize,
        charge_range: ChargeRange,
    ) -> Box<dyn ChargeIterator> {
        match self {
            ChargeStrategy::ChargeRange => {
                Box::from(ChargeRangeIter::new(charge_range.0, charge_range.1))
            }
            ChargeStrategy::QuickCharge => Box::new(quick_charge_w(peaks, position, charge_range)),
        }
    }
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
            if line == "\n" {
                assert_eq!(charge_list, Vec::<i32>::new());
            } else {
                let expected: Vec<_> = line
                    .strip_suffix("\n")
                    .unwrap()
                    .split(',')
                    .map(|t| t.parse::<i32>().unwrap())
                    .collect();
                assert_eq!(charge_list, expected);
            }
        }

        Ok(())
    }
}
