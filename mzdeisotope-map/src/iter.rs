use mzpeaks::{
    feature::{Feature, TimeInterval},
    prelude::*,
    CentroidPeak, MZ,
};

#[derive(Debug)]
pub struct FeatureSetIter<'a, Y> {
    features: &'a [Option<&'a Feature<MZ, Y>>],
    pub start_time: f64,
    pub end_time: f64,
    pub last_time_seen: f64,
    index_list: Vec<usize>,
}

impl<Y> Iterator for FeatureSetIter<'_, Y> {
    type Item = (f64, Vec<Option<CentroidPeak>>);

    fn next(&mut self) -> Option<Self::Item> {
        self.get_next_value()
    }
}

macro_rules! f_at {
    ($f:expr, $at:expr) => {
        $f.at($at)
    };
}

impl<'a, Y> FeatureSetIter<'a, Y> {
    pub fn new_with_time_interval(features: &'a [Option<&'a Feature<MZ, Y>>], start_time: f64, end_time: f64) -> Self {
        let n = features.len();
        let index_list = (0..n).map(|_| 0).collect();

        let mut this = Self {
            features,
            start_time,
            end_time,
            index_list,
            last_time_seen: f64::NEG_INFINITY,
        };
        this.initialize_indices();
        this
    }

    pub fn new(features: &'a [Option<&'a Feature<MZ, Y>>]) -> Self {
        let mut start_time: f64 = 0.0;
        let mut end_time: f64 = f64::INFINITY;

        for f in features.iter().flatten() {
            {
                if let Some(t) = f.start_time() {
                    if start_time < t {
                        start_time = t;
                    }
                }
                if let Some(t) = f.end_time() {
                    if end_time > t {
                        end_time = t;
                    }
                }
            }
        }

        if end_time < start_time {
            std::mem::swap(&mut start_time, &mut end_time);
        }

        Self::new_with_time_interval(features, start_time, end_time)
    }

    fn get_next_time(&self) -> Option<f64> {
        let mut time = f64::INFINITY;

        for (i, f) in self.features.iter().enumerate() {
            if let Some(f) = f {
                let ix = self.index_list[i];
                if ix < f.len() {
                    let ix_time = f_at!(f, ix).unwrap().1;
                    if ix_time < time && (ix_time <= self.end_time || ix_time.is_close(&self.end_time)) && ix_time > self.last_time_seen {
                        time = ix_time;
                    }
                }
            }
        }

        if time.is_infinite() {
            return None;
        }
        Some(time)
    }

    fn has_more(&self) -> bool {
        let mut j = 0;
        let n = self.features.len();

        for (i, f) in self.features.iter().enumerate() {
            if let Some(f) = f {
                let ix = self.index_list[i];
                let done = ix >= f.len();
                let done = if !done {
                    let time_at = f_at!(f, ix).unwrap().1;
                    time_at > self.end_time
                } else {
                    true
                };
                j += done as usize;
            } else {
                j += 1;
            }
        }
        j != n
    }

    fn initialize_indices(&mut self) {
        self.features.iter().enumerate().for_each(|(i, f)| {
            if let Some(f) = f {
                let (ix, _) = f.find_time(self.start_time);
                self.index_list[i] = ix.unwrap();
            } else {
                self.index_list[i] = 0;
            }
        })
    }

    fn get_peaks_for_time(&self, time: f64) -> Vec<Option<CentroidPeak>> {
        let mut peaks = Vec::new();
        for f in self.features.iter() {
            if let Some(f) = f {
                if !f.is_empty() {
                    let (ix, err) = f.find_time(time);
                    if err.abs() > 1e-3 {
                        peaks.push(None);
                        continue;
                    }
                    if let Some(ix) = ix {
                        let p = f.at(ix).unwrap();
                        peaks.push(Some(CentroidPeak::new(p.0, p.2, ix as u32)));
                    }
                } else {
                    peaks.push(None)
                }
            } else {
                peaks.push(None)
            }
        }
        peaks
    }

    fn get_next_value(&mut self) -> Option<(f64, Vec<Option<CentroidPeak>>)> {
        if !self.has_more() {
            return None
        }
        let time = self.get_next_time();
        if let Some(time) = time {
            let peaks = self.get_peaks_for_time(time);
            for (i, p) in peaks.iter().enumerate() {
                if let Some(p) = p {
                    if p.index as usize >= self.index_list[i] {
                        self.index_list[i] += 1;
                    }
                }
            }
            self.last_time_seen = time;
            Some((time, peaks))
        } else {
            None
        }
    }
}
