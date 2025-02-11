/*! Description of isotopic pattern fits */
use std::cmp::Ordering;

use chemical_elements::isotopic_pattern::TheoreticalIsotopicPattern;

use mzsignal::smooth::moving_average;
use mzdeisotope::scorer::ScoreType;
use mzpeaks::{feature::TimeInterval, feature_map::FeatureMapLike, CoordinateRange};

use crate::traits::FeatureMapType;

#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
pub struct MapCoordinate {
    pub coord: f64,
    pub time: f64,
}

impl MapCoordinate {
    pub fn new(coord: f64, time: f64) -> Self {
        Self { coord, time }
    }
}

#[derive(Debug, Clone)]
pub struct FeatureSetFit {
    pub features: Vec<Option<usize>>,
    pub start: MapCoordinate,
    pub end: MapCoordinate,
    pub score: ScoreType,
    pub theoretical: TheoreticalIsotopicPattern,
    pub mz: f64,
    pub neutral_mass: f64,
    pub charge: i32,
    pub missing_features: usize,
    pub n_points: usize,
    pub monoisotopic_index: usize,
    pub scores: Vec<ScoreType>,
    pub times: Vec<f64>,
}

impl PartialOrd for FeatureSetFit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FeatureSetFit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score)
    }
}

impl PartialEq for FeatureSetFit {
    fn eq(&self, other: &Self) -> bool {
        self.features == other.features
            && self.score == other.score
            && self.theoretical == other.theoretical
            && self.charge == other.charge
    }
}

impl Eq for FeatureSetFit {}

impl FeatureSetFit {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        features: Vec<Option<usize>>,
        theoretical: TheoreticalIsotopicPattern,
        start: MapCoordinate,
        end: MapCoordinate,
        score: ScoreType,
        charge: i32,
        missing_features: usize,
        neutral_mass: f64,
        n_points: usize,
        scores: Vec<ScoreType>,
        times: Vec<f64>,
    ) -> Self {
        let mz = theoretical.peaks.first().map(|p| p.mz).unwrap();
        Self {
            features,
            theoretical,
            start,
            end,
            score,
            charge,
            missing_features,
            monoisotopic_index: 0,
            mz,
            neutral_mass,
            n_points,
            scores,
            times,
        }
    }

    pub fn count_null_features(&self) -> usize {
        self.features.iter().map(|f| f.is_none() as usize).sum()
    }

    pub fn has_multiple_real_features(&self) -> bool {
        (self.features.len() - self.count_null_features()) > 1
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn find_bounds<Y: Clone>(&self, feature_map: &FeatureMapType<Y>, detection_threshold: f32) -> CoordinateRange<Y> {
        let mut start_time: f64 = f64::INFINITY;
        let mut end_time: f64 = f64::INFINITY;

        for (f, p) in self.features.iter().zip(self.theoretical.iter()) {
            if let Some(i) = f {
                let f = feature_map.get_item(*i);
                if p.intensity() >= detection_threshold {
                    start_time = f.start_time().unwrap().min(start_time);
                    end_time = f.end_time().unwrap().min(end_time);
                }

            } else {
                continue;
            }
        }

        CoordinateRange::new(Some(start_time), Some(end_time))
    }

    pub fn find_separation<Y: Clone>(&self, feature_map: &FeatureMapType<Y>, detection_threshold: f32) -> (CoordinateRange<Y>, Vec<ScoreSegment>) {
        let mut time_range = self.find_bounds(feature_map, detection_threshold);
        if self.n_points > 0 {
            let mut segments = Vec::new();
            let mut last_score = ScoreType::INFINITY;
            let mut scores = Vec::with_capacity(self.scores.len());
            scores.resize(self.scores.len(), 0.0);
            moving_average::<f32, 1>(&self.scores, &mut scores);
            let mut begin_i = 0;
            for (i, score) in scores.iter().copied().enumerate() {
                if score > 0.0 && last_score < 0.0 {
                    begin_i = i
                } else if score < 0.0 && last_score > 0.0 {
                    let end_i = i;
                    segments.push(ScoreSegment::new(
                        begin_i, end_i,
                        scores[begin_i..end_i].iter().copied().sum::<f32>()
                    ));
                    begin_i = i;
                }
                last_score = score;
            }
            let end_i = scores.len().saturating_sub(1);
            segments.push(ScoreSegment::new(
                begin_i,
                end_i,
                scores[begin_i..end_i].iter().sum()
            ));
            let segment = segments.iter().max_by(|a, b| {a.score.total_cmp(&b.score)}).copied().unwrap();
            if time_range.start.unwrap() < self.times[segment.start] && segment.start != 0 {
                time_range.start = Some(self.times[segment.start]);
            }
            if time_range.end.unwrap() > self.times[segment.end] {
                time_range.end = Some(self.times[segment.end])
            }
            (time_range, segments)
        } else {
            (time_range, Vec::new())
        }
    }
}


#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct ScoreSegment {
    pub start: usize,
    pub end: usize,
    pub score: f32
}

impl ScoreSegment {
    pub fn new(start: usize, end: usize, score: f32) -> Self {
        Self { start, end, score }
    }
}
