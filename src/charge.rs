use std::cmp;

#[derive(Debug, Clone)]
pub struct ChargeIterator {
    min: i32,
    max: i32,
    sign: i8,
    index: usize,
    size: usize,
}

impl ChargeIterator {
    pub fn new(min: i32, max: i32) -> ChargeIterator {
        let low = cmp::min(min.abs(), max.abs());
        let high = cmp::max(min.abs(), max.abs());
        let sign = (min / min.abs()) as i8;
        let size = (high - low) as usize;
        ChargeIterator {
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

impl Iterator for ChargeIterator {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        self.next_charge()
    }
}

impl From<(i32, i32)> for ChargeIterator {
    fn from(pair: (i32, i32)) -> ChargeIterator {
        ChargeIterator::new(pair.0, pair.1)
    }
}
