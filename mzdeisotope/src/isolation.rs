use mzdata::spectrum::IsolationWindow;
use mzpeaks::{prelude::*, MZPeakSetType, MassPeakSetType};

use mzpeaks::coordinate::{SimpleInterval, Span1D};

use crate::solution::DeconvolvedSolutionPeak;

#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
pub struct Coisolation {
    pub neutral_mass: f64,
    pub intensity: f32,
    pub charge: Option<i32>,
}

impl Coisolation {
    pub fn new(neutral_mass: f64, intensity: f32, charge: Option<i32>) -> Self {
        Self {
            neutral_mass,
            intensity,
            charge,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrecursorPurityEstimator {
    pub lower_extension: f64,
    pub default_width: f64,
}

impl Default for PrecursorPurityEstimator {
    fn default() -> Self {
        Self {
            lower_extension: 1.5,
            default_width: 1.5,
        }
    }
}

impl PrecursorPurityEstimator {
    pub fn new(lower_extension: f64, default_width: f64) -> Self {
        Self {
            lower_extension,
            default_width,
        }
    }

    fn infer_isolation_interval(
        &self,
        precursor_peak: &DeconvolvedSolutionPeak,
        isolation_window: Option<&IsolationWindow>,
    ) -> SimpleInterval<f64> {
        match isolation_window {
            Some(window) => {
                if window.lower_bound == 0.0 {
                    SimpleInterval::new(
                        precursor_peak.mz() - self.default_width,
                        precursor_peak.mz() + self.default_width,
                    )
                } else {
                    SimpleInterval::new(window.lower_bound as f64, window.upper_bound as f64)
                }
            }
            None => SimpleInterval::new(
                precursor_peak.mz() - self.default_width,
                precursor_peak.mz() + self.default_width,
            ),
        }
    }

    pub fn coisolation(
        &self,
        peaks: &MassPeakSetType<DeconvolvedSolutionPeak>,
        precursor_peak: &DeconvolvedSolutionPeak,
        isolation_window: Option<&IsolationWindow>,
        relative_intensity_threshold: f32,
        ignore_singly_charged: bool,
    ) -> Vec<Coisolation> {
        let mut isolation_window = self.infer_isolation_interval(precursor_peak, isolation_window);
        isolation_window.start -= self.lower_extension;
        let intensity_threshold = precursor_peak.intensity * relative_intensity_threshold;
        let isolates: Vec<_> = peaks
            .iter()
            .filter(|p| {
                isolation_window.contains(&p.mz())
                    && p.intensity > intensity_threshold
                    && !precursor_peak.eq(p)
                    && (!ignore_singly_charged || (p.charge.abs() > 1))
            })
            .map(|p| Coisolation::new(p.neutral_mass, p.intensity, Some(p.charge)))
            .collect();
        isolates
    }

    pub fn precursor_purity<C: CentroidLike>(
        &self,
        peaks: &MZPeakSetType<C>,
        precursor_peak: &DeconvolvedSolutionPeak,
        isolation_window: Option<&IsolationWindow>,
    ) -> f32 {
        let isolation_window = self.infer_isolation_interval(precursor_peak, isolation_window);
        let assigned: f32 = precursor_peak.envelope.iter().map(|f| f.intensity).sum();
        let total: f32 = peaks
            .between(
                isolation_window.start,
                isolation_window.end,
                Tolerance::PPM(5.0),
            )
            .iter()
            .map(|p| p.intensity())
            .sum();
        if total == 0.0 {
            0.0
        } else {
            assigned / total
        }
    }
}
