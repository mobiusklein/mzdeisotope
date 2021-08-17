use chemical_elements::isotopic_pattern::{Peak as TheoreticalPeak, TheoreticalIsotopicPattern};
use mzpeaks::prelude::*;

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
    ) -> f64;
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
    ) -> f64 {
        let mass_error = (experimental.mz() - theoretical.mz()).abs();
        if mass_error > self.error_tolerance {
            return 0.0;
        }
        let mass_accuracy = 1.0 - (mass_error / self.error_tolerance);

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

        let score = theoretical.intensity().sqrt() as f64 * mass_accuracy * abundance_diff as f64;
        score
    }

    #[inline]
    pub fn score<C: CentroidLike>(
        &self,
        experimental: &Vec<C>,
        theoretical: &TheoreticalIsotopicPattern,
    ) -> f64 {
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
    ) -> f64 {
        MSDeconvScorer::score(self, experimental, theoretical)
    }
}
