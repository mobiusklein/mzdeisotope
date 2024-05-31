use identity_hash::BuildIdentityHasher;
use itertools::Itertools;

use mzdeisotope::scorer::ScoreInterpretation;

// use super::fit::{FitKey, FitNode, FitNodeGraphInner, FitRef};
use super::fit::{FitKey, FitNode, FitRef};
use crate::feature_fit::MapCoordinate;

pub type SubgraphSolution = Vec<FitRef>;

#[derive(Debug, Clone)]
pub struct DependenceCluster {
    pub dependencies: Vec<FitRef>,
    pub score_ordering: ScoreInterpretation,
    pub start: MapCoordinate,
    pub end: MapCoordinate,
}


impl DependenceCluster {
    pub fn new(
        dependencies: Vec<FitRef>,
        score_ordering: ScoreInterpretation,
    ) -> DependenceCluster {
        let mut cluster = DependenceCluster {
            dependencies,
            score_ordering,
            start: MapCoordinate::default(),
            end: MapCoordinate::default(),
        };
        cluster.reset();
        cluster
    }

    pub fn len(&self) -> usize {
        self.dependencies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dependencies.is_empty()
    }

    pub fn contains(&self, fit: &FitRef) -> bool {
        self.dependencies.iter().any(|f| f == fit)
    }

    pub fn bounds(&self) -> (MapCoordinate, MapCoordinate) {
        let mut start = MapCoordinate::new(f64::INFINITY, f64::INFINITY);
        let mut end = MapCoordinate::new(-f64::INFINITY, -f64::INFINITY);

        for f in self.dependencies.iter() {
            if start.coord > f.start.coord {
                start.coord = f.start.coord;
            }
            if start.time > f.start.time {
                start.time = f.start.time;
            }

            if end.coord < f.end.coord {
                end.coord = f.end.coord;
            }
            if end.time < f.end.time {
                end.time = f.end.time;
            }
        }
        (start, end)
    }

    fn reset(&mut self) {
        match self.score_ordering {
            ScoreInterpretation::HigherIsBetter => {
                self.dependencies
                    .sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap().reverse());
            }
            ScoreInterpretation::LowerIsBetter => {
                self.dependencies
                    .sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
            }
        }
        (self.start, self.end) = self.bounds();
    }

    pub fn best_fit(&self) -> Option<&FitRef> {
        self.dependencies.first()
    }

    pub fn add(&mut self, fit: FitRef) {
        self.dependencies.push(fit);
        self.reset()
    }

    pub fn iter(&self) -> std::slice::Iter<FitRef> {
        self.dependencies.iter()
    }
}