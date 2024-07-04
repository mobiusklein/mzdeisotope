//! Isotopic pattern evaluation tools
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
        experimental: &[C],
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType;
    fn interpretation(&self) -> ScoreInterpretation {
        ScoreInterpretation::HigherIsBetter
    }
}

/// An implementation of the scoring function used in [MS-Deconv][1]
///
///
/// ```math
/// \begin{split}
///     s_{mz}(e, t) &= \begin{cases}
///         1 - \frac{\|mz(e) - mz(t)\|}{d} & \text{if } \|mz(e) - mz(t)\| < d,\\
///         0 & \text{otherwise}
///     \end{cases}\\
///
///     s_{int}(e, t) &= \begin{cases}
///         1 - \frac{int(t) - int(e)}{int(e)} & \text{if } int(e) < int(t) \text{ and } \frac{int(t) - int(e)}{int(e)} \le 1, \\
///         \sqrt{1 - \frac{int(e) - int(t)}{int(t)}} & \text{if } int(e) \ge int(t) \text{ and } \frac{int(e) - int(t)}{int(t)} \le 1,\\
///         0 & \text{otherwise}
///     \end{cases}\\
///
///     \text{S}(e, t) &= \sqrt{int(t)}\times s_{mz}(e,t) \times s_{int}(e, t)
/// \end{split}
/// ```
/// # References
/// - [1]: <https://doi.org/10.1074/mcp.M110.002766>
///     Liu, X., Inbar, Y., Dorrestein, P. C., Wynne, C., Edwards, N., Souda, P., …
///     Pevzner, P. A. (2010). Deconvolution and database search of complex tandem
///     mass spectra of intact proteins: a combinatorial approach. Molecular & Cellular
///     Proteomics : MCP, 9(12), 2772–2782.
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
        theoretical.intensity().sqrt() * mass_accuracy * abundance_diff
    }

    #[inline]
    pub fn score<C: CentroidLike>(
        &self,
        experimental: &[C],
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
    #[inline]
    fn score<C: CentroidLike>(
        &self,
        experimental: &[C],
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType {
        MSDeconvScorer::score(self, experimental, theoretical)
    }
}

/// Evaluate an isotopic fit using a [G-test](https://en.wikipedia.org/wiki/G-test)
///
/// ```math
/// G = 2 \displaystyle\sum_i^n {o_i * (\log o_i  - \log e_i)}
/// ```
///
/// where $`o_i`$ is the intensity of the ith experimental peak
/// and $`e_i`$ is the intensity of the ith theoretical peak.
///
/// The G statistic is on the scale of the signal used.
#[derive(Default, Debug, Clone, Copy)]
pub struct GTestScorer {}

impl GTestScorer {
    #[inline]
    pub fn score<C: CentroidLike>(
        &self,
        experimental: &[C],
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType {
        2.0 * experimental
            .iter()
            .zip(theoretical.iter())
            .map(|(o, e)| {
                let oi = o.intensity();
                let ei = e.intensity();
                (oi * (oi.ln() - ei.ln())) as ScoreType
            })
            .sum::<ScoreType>()
    }
}

impl IsotopicPatternScorer for GTestScorer {
    fn score<C: CentroidLike>(
        &self,
        experimental: &[C],
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType {
        GTestScorer::score(self, experimental, theoretical)
    }

    fn interpretation(&self) -> ScoreInterpretation {
        ScoreInterpretation::LowerIsBetter
    }
}


/// Evaluate an isotopic fit using a [G-test](https://en.wikipedia.org/wiki/G-test), after normalizing the
/// list of experimental and theoretical peaks to both sum to 1.
/// ```math
/// G = 2 \displaystyle\sum_i^n {o_i * (\log o_i  - \log e_i)}
/// ```
/// where $`o_i`$ is the intensity of the ith experimental peak and $`e_i`$ is the
/// intensity of the ith theoretical peak.
#[derive(Default, Debug, Clone, Copy)]
pub struct ScaledGTestScorer {}

impl ScaledGTestScorer {
    #[inline]
    pub fn score<C: CentroidLike>(
        &self,
        experimental: &[C],
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType {
        let total_o: f32 = experimental.iter().map(|p| p.intensity()).sum();
        let total_e: f32 = theoretical.iter().map(|p| p.intensity()).sum();
        2.0 * experimental
            .iter()
            .zip(theoretical.iter())
            .map(|(o, e)| {
                let oi = o.intensity() / total_o;
                let ei = e.intensity() / total_e;
                (oi * (oi.ln() - ei.ln())) as ScoreType
            })
            .sum::<ScoreType>()
    }
}

impl IsotopicPatternScorer for ScaledGTestScorer {
    #[inline]
    fn score<C: CentroidLike>(
        &self,
        experimental: &[C],
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType {
        ScaledGTestScorer::score(self, experimental, theoretical)
    }

    fn interpretation(&self) -> ScoreInterpretation {
        ScoreInterpretation::LowerIsBetter
    }
}


/// Combines [`MSDeconvScorer`] with a penalty of [`ScaledGTestScorer`].
///
/// ```math
/// S(e, t) = \operatorname{MS-Deconv}(e, t) \times (1 - \mathit{w} \operatorname{G-test}(e, t))
/// ```
/// where $`w`$ is [`PenalizedMSDeconvScorer::penalty_factor`]
#[derive(Debug, Clone, Copy)]
pub struct PenalizedMSDeconvScorer {
    msdeconv: MSDeconvScorer,
    penalizer: ScaledGTestScorer,
    /// The scaling of the normalized G-statistic to apply. Smaller
    /// values lead to less penalty, where `0` makes this equivalent to [`MSDeconvScorer`]
    pub penalty_factor: ScoreType,
}

impl IsotopicPatternScorer for PenalizedMSDeconvScorer {
    #[inline]
    fn score<C: CentroidLike>(
        &self,
        experimental: &[C],
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType {
        PenalizedMSDeconvScorer::score(self, experimental, theoretical)
    }
}

impl Default for PenalizedMSDeconvScorer {
    fn default() -> Self {
        Self {
            msdeconv: Default::default(),
            penalizer: Default::default(),
            penalty_factor: 2.0,
        }
    }
}

impl PenalizedMSDeconvScorer {
    pub fn new(error_tolerance: f64, penalty_factor: ScoreType) -> Self {
        Self {
            msdeconv: MSDeconvScorer::new(error_tolerance),
            penalizer: ScaledGTestScorer::default(),
            penalty_factor,
        }
    }

    #[inline]
    pub fn score<C: CentroidLike>(
        &self,
        experimental: &[C],
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType {
        let base_score = self.msdeconv.score(experimental, theoretical);
        let penalty = self.penalizer.score(experimental, theoretical);
        base_score - (self.penalty_factor * penalty)
    }
}

/// A type that sort and filter [`IsotopicFit`] instances.
pub trait IsotopicFitFilter {
    /// Filter out fits which do not satisfy the filter.
    fn filter(&self, mut fits: HashSet<IsotopicFit>) -> HashSet<IsotopicFit> {
        fits.retain(|f| self.test(f));
        fits
    }

    /// Get the best solution from an iterator
    fn select<I: Iterator<Item = IsotopicFit>>(&self, fits: I) -> Option<IsotopicFit>;

    /// Test if an [`IsotopicFit`] passes the required score threshold, though
    /// it may perform other checks in addition to testing the score.
    fn test(&self, fit: &IsotopicFit) -> bool {
        self.test_score(fit.score)
    }

    /// Test if a score is good enough to satisfy the filter.
    fn test_score(&self, fit: ScoreType) -> bool;
}


/// A [`IsotopicFitFilter`] that has a minimum score threshold and
/// prefers larger scores. The default form uses a threshold of `0.0`.
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
    #[inline]
    fn test(&self, fit: &IsotopicFit) -> bool {
        fit.score >= self.threshold
    }

    #[inline]
    fn test_score(&self, fit: ScoreType) -> bool {
        fit >= self.threshold
    }

    fn select<I: Iterator<Item = IsotopicFit>>(&self, fits: I) -> Option<IsotopicFit> {
        fits.max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .and_then(|f| if self.test(&f) { Some(f) } else { None })
    }
}

/// A [`IsotopicFitFilter`] that has a maximum score threshold and
/// prefers smaller scores. The default form uses a threshold of `1.0`.
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
    #[inline]
    fn test(&self, fit: &IsotopicFit) -> bool {
        self.threshold >= fit.score
    }

    #[inline]
    fn test_score(&self, fit: ScoreType) -> bool {
        self.threshold >= fit
    }

    fn select<I: Iterator<Item = IsotopicFit>>(&self, fits: I) -> Option<IsotopicFit> {
        fits.min_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .and_then(|f| if self.test(&f) { Some(f) } else { None })
    }
}

#[cfg(test)]
mod test {

    use chemical_elements::isotopic_pattern::{Peak, TheoreticalIsotopicPattern};
    use mzpeaks::CentroidPeak;

    use super::{GTestScorer, MSDeconvScorer, ScaledGTestScorer};

    fn make_experimental_peaks() -> Vec<CentroidPeak> {
        vec![
            CentroidPeak::new(739.920, 8356.829, 30),
            CentroidPeak::new(740.255, 8006.456, 31),
            CentroidPeak::new(740.589, 4970.605, 32),
            CentroidPeak::new(740.923, 2215.961, 33),
        ]
    }

    fn make_theoretical() -> TheoreticalIsotopicPattern {
        TheoreticalIsotopicPattern::new(
            vec![
                Peak {
                    mz: 739.920306,
                    intensity: 8310.933747,
                    charge: -3,
                },
                Peak {
                    mz: 740.254733,
                    intensity: 8061.025466,
                    charge: -3,
                },
                Peak {
                    mz: 740.588994,
                    intensity: 4926.998052,
                    charge: -3,
                },
                Peak {
                    mz: 740.923235,
                    intensity: 2250.893651,
                    charge: -3,
                },
            ],
            739.920306,
        )
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
