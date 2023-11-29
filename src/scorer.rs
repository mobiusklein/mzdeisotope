use std::collections::HashSet;

use chemical_elements::isotopic_pattern::{Peak as TheoreticalPeak, TheoreticalIsotopicPattern};
use mzpeaks::prelude::*;

use crate::isotopic_fit::IsotopicFit;

pub type ScoreType = f32;


#[derive(Debug, Clone, Copy)]
pub enum ScoreInterpretation {
    HigherIsBetter,
    LowerIsBetter,
}

pub trait IsotopicPatternScorer {
    fn score<C: CentroidLike>(
        &self,
        experimental: &Vec<C>,
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType;
    fn interpretation(&self) -> ScoreInterpretation {
        ScoreInterpretation::HigherIsBetter
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MSDeconvScorer {
    pub error_tolerance: f64,
}

impl MSDeconvScorer {
    #[inline]
    pub fn score_peak<C: CentroidLike>(
        &self,
        experimental: &C,
        theoretical: &TheoreticalPeak,
    ) -> ScoreType {
        let mass_error = (experimental.mz() - theoretical.mz()).abs();
        if mass_error > self.error_tolerance {
            return 0.0;
        }
        let mass_accuracy = 1.0 - (mass_error / self.error_tolerance) as ScoreType;

        let ratio = (theoretical.intensity() - experimental.intensity()) / experimental.intensity();

        let abundance_diff = if experimental.intensity() < theoretical.intensity() && ratio <= 1.0 {
            1.0 - ratio
        } else if experimental.intensity() >= theoretical.intensity()
            && (experimental.intensity() - theoretical.intensity()) / experimental.intensity()
                <= 1.0
        {
            1.0 - (experimental.intensity() - theoretical.intensity()) / experimental.intensity()
        } else {
            return 0.0;
        };

        
        theoretical.intensity().sqrt() as ScoreType * mass_accuracy * abundance_diff as ScoreType
    }

    #[inline]
    pub fn score<C: CentroidLike>(
        &self,
        experimental: &Vec<C>,
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType {
        experimental
            .iter()
            .zip(theoretical.iter())
            .map(|(e, t)| self.score_peak(e, t))
            .sum()
    }

    pub fn new(error_tolerance: f64) -> Self {
        Self { error_tolerance }
    }
}

impl Default for MSDeconvScorer {
    fn default() -> MSDeconvScorer {
        MSDeconvScorer {
            error_tolerance: 0.02,
        }
    }
}

impl IsotopicPatternScorer for MSDeconvScorer {
    fn score<C: CentroidLike>(
        &self,
        experimental: &Vec<C>,
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType {
        MSDeconvScorer::score(self, experimental, theoretical)
    }
}


#[derive(Default, Debug, Clone, Copy)]
pub struct GTestScorer {}


impl GTestScorer {
    pub fn score<C: CentroidLike>(
        &self,
        experimental: &Vec<C>,
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType {
        2.0 * experimental.iter().zip(theoretical.iter()).map(|(o, e)| {
            let oi = o.intensity();
            let ei = e.intensity();
            (oi * (oi.ln() - ei.ln())) as ScoreType
        }).sum::<ScoreType>()
    }
}


impl IsotopicPatternScorer for GTestScorer {
    fn score<C: CentroidLike>(
        &self,
        experimental: &Vec<C>,
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType {
        GTestScorer::score(self, experimental, theoretical)
    }

    fn interpretation(&self) -> ScoreInterpretation {
        ScoreInterpretation::LowerIsBetter
    }
}


#[derive(Default, Debug, Clone, Copy)]
pub struct ScaledGTestScorer {}


impl ScaledGTestScorer {
    pub fn score<C: CentroidLike>(
        &self,
        experimental: &Vec<C>,
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType {
        let total_o: f32 = experimental.iter().map(|p| p.intensity()).sum();
        let total_e: f32 = theoretical.iter().map(|p| p.intensity()).sum();
        2.0 * experimental.iter().zip(theoretical.iter()).map(|(o, e)| {
            let oi = o.intensity() / total_o;
            let ei = e.intensity() / total_e;
            (oi * (oi.ln() - ei.ln())) as ScoreType
        }).sum::<ScoreType>()
    }
}


impl IsotopicPatternScorer for ScaledGTestScorer {
    fn score<C: CentroidLike>(
        &self,
        experimental: &Vec<C>,
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType {
        ScaledGTestScorer::score(self, experimental, theoretical)
    }

    fn interpretation(&self) -> ScoreInterpretation {
        ScoreInterpretation::LowerIsBetter
    }
}


#[derive(Debug, Clone, Copy)]
pub struct PenalizedMSDeconvScorer {
    msdeconv: MSDeconvScorer,
    penalizer: ScaledGTestScorer,
    penalty_factor: ScoreType
}

impl IsotopicPatternScorer for PenalizedMSDeconvScorer {
    fn score<C: CentroidLike>(
        &self,
        experimental: &Vec<C>,
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType {
        PenalizedMSDeconvScorer::score(self, experimental, theoretical)
    }
}

impl Default for PenalizedMSDeconvScorer {
    fn default() -> Self {
        Self { msdeconv: Default::default(), penalizer: Default::default(), penalty_factor: 2.0 }
    }
}

impl PenalizedMSDeconvScorer {
    pub fn new(error_tolerance: f64, penalty_factor: ScoreType) -> Self {
        Self {
            msdeconv: MSDeconvScorer::new(error_tolerance),
            penalizer: ScaledGTestScorer::default(),
            penalty_factor
        }
    }

    pub fn score<C: CentroidLike>(
        &self,
        experimental: &Vec<C>,
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType {
        let base_score = self.msdeconv.score(experimental, theoretical);
        let penalty = self.penalizer.score(experimental, theoretical);
        base_score - (self.penalty_factor * penalty)
    }
}

pub trait IsotopicFitFilter {
    fn filter(&self, mut fits: HashSet<IsotopicFit>) -> HashSet<IsotopicFit> {
        fits.retain(|f| self.test(f));
        fits
    }
    fn test(&self, fit: &IsotopicFit) -> bool;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MaximizingFitFilter {
    pub threshold: ScoreType,
}

impl MaximizingFitFilter {
    pub fn new(threshold: ScoreType) -> Self {
        Self { threshold }
    }
}

impl IsotopicFitFilter for MaximizingFitFilter {
    fn test(&self, fit: &IsotopicFit) -> bool {
        fit.score >= self.threshold
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MinimizingFitFilter {
    pub threshold: ScoreType,
}

impl MinimizingFitFilter {
    pub fn new(threshold: ScoreType) -> Self {
        Self { threshold }
    }
}

impl Default for MinimizingFitFilter {
    fn default() -> Self {
        Self { threshold: 1.0 }
    }
}

impl IsotopicFitFilter for MinimizingFitFilter {
    fn test(&self, fit: &IsotopicFit) -> bool {
        self.threshold >= fit.score
    }
}


#[cfg(test)]
mod test {

    use mzpeaks::CentroidPeak;
    use chemical_elements::isotopic_pattern::{Peak, TheoreticalIsotopicPattern};

    use super::{MSDeconvScorer, GTestScorer, ScaledGTestScorer};

    fn make_experimental_peaks() -> Vec<CentroidPeak> {
        vec![
            CentroidPeak::new(739.920, 8356.829, 30),
            CentroidPeak::new(740.255, 8006.456, 31),
            CentroidPeak::new(740.589, 4970.605, 32),
            CentroidPeak::new(740.923, 2215.961, 33),
        ]
    }

    fn make_theoretical() -> TheoreticalIsotopicPattern {
        TheoreticalIsotopicPattern::new(vec![
            Peak { mz: 739.920306, intensity: 8310.933747, charge: -3 },
            Peak {
                mz: 740.254733,
                intensity: 8061.025466,
                charge: -3
            },
            Peak {
                mz: 740.588994,
                intensity: 4926.998052,
                charge: -3
            },
            Peak {
                mz: 740.923235,
                intensity: 2250.893651,
                charge: -3
            },
        ], 739.920306)
    }

    #[test]
    fn test_msdeconv() {
        let eid = make_experimental_peaks();
        let tid = make_theoretical();

        let scorer = MSDeconvScorer::new(0.02);
        let score = scorer.score(&eid, &tid);
        assert!((score - 292.960_27).abs() < 1e-3);
    }

    #[test]
    fn test_gtest() {
        let eid = make_experimental_peaks();
        let tid = make_theoretical();

        let scorer = GTestScorer::default();
        let score = scorer.score(&eid, &tid);
        assert!((score - 1.556_16).abs() < 1e-3);
    }

    #[test]
    fn test_scaled_gtest() {
        let eid = make_experimental_peaks();
        let tid = make_theoretical();

        let scorer = ScaledGTestScorer::default();
        let score = scorer.score(&eid, &tid);
        assert!((score - 6.593_764e-5).abs() < 1e-3);
    }
}