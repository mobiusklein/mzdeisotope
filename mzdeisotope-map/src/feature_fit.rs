/*! Description of isotopic pattern fits */
use std::cmp::Ordering;

use chemical_elements::isotopic_pattern::TheoreticalIsotopicPattern;

use mzdeisotope::scorer::ScoreType;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
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
        Some(self.score.total_cmp(&other.score))
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
}
