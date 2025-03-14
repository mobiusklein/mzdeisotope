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
    has_more: bool,
}

impl<Y> Iterator for FeatureSetIter<'_, Y> {
    type Item = (f64, Vec<Option<CentroidPeak>>);

    fn next(&mut self) -> Option<Self::Item> {
        self.get_next_value()
    }
}

impl<'a, Y> FeatureSetIter<'a, Y> {
    pub fn new_with_time_interval(
        features: &'a [Option<&'a Feature<MZ, Y>>],
        start_time: f64,
        end_time: f64,
    ) -> Self {
        let n = features.len();
        let index_list = (0..n).map(|_| 0).collect();

        let mut this = Self {
            features,
            start_time,
            end_time,
            index_list,
            last_time_seen: f64::NEG_INFINITY,
            has_more: true,
        };
        this.initialize_indices();
        this.has_more = this.has_more();
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

        for (f, ix) in self.features.iter().zip(self.index_list.iter().copied()) {
            if let Some(f) = f {
                if let Some(ix_time) = f.time_view().get(ix).copied() {
                    if ix_time < time
                        && (ix_time <= self.end_time || ix_time.is_close(&self.end_time))
                        && ix_time > self.last_time_seen
                    {
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

        for (f, ix) in self.features.iter().zip(self.index_list.iter().copied()) {
            if let Some(f) = f {
                let done = ix >= f.len();
                let done = if !done {
                    let time_at = f.time_view()[ix];
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

    fn get_peaks_for_next_time(&mut self, time: f64) -> Vec<Option<CentroidPeak>> {
        let mut peaks = Vec::new();
        for (f, i) in self.features.iter().zip(&mut self.index_list) {
            if let Some(f) = f {
                if *i >= f.len() {
                    peaks.push(None);
                    continue;
                }
                let (mz, time_at, intensity) = f.at(*i).unwrap();
                let time_err = time_at - time;
                if time_err.abs() > 1e-3 {
                    peaks.push(None);
                } else {
                    peaks.push(Some(CentroidPeak::new(mz, intensity, *i as u32)));
                    *i += 1;
                }
            } else {
                peaks.push(None);
            }
        }
        peaks
    }

    fn get_next_value(&mut self) -> Option<(f64, Vec<Option<CentroidPeak>>)> {
        if !self.has_more {
            return None;
        }
        let time = self.get_next_time();
        if let Some(time) = time {
            let peaks = self.get_peaks_for_next_time(time);
            self.last_time_seen = time;
            Some((time, peaks))
        } else {
            self.has_more = false;
            None
        }
    }
}
