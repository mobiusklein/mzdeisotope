//! Tools for evaluating isolation windows

use mzdata::spectrum::IsolationWindow;
use mzpeaks::{prelude::*, MZPeakSetType, MassPeakSetType};

use mzpeaks::coordinate::{SimpleInterval, Span1D};

use crate::solution::DeconvolvedSolutionPeak;

/// A precursor ion co-isolated in an isolation window
#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
pub struct Coisolation {
    /// The estimated neutral mass of the ion
    pub neutral_mass: f64,
    /// The total intensity of the ion's isotopic pattern
    pub intensity: f32,
    /// The estimated charge of the ion
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


/// An estimator of precursor selection purity
#[derive(Debug, Clone)]
pub struct PrecursorPurityEstimator {
    /// How far beyond the lower bound of the isolation window to search for
    /// co-isolating peaks
    pub lower_extension: f64,
    /// The default width assumed for an isolation window when a width is not
    /// provided
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

    /// Find all co-isolating ions around a precursor peak, given an isolation window.
    ///
    /// # Arguments
    /// - `peaks`: The deconvolved peak set itself to search for coisolations in
    /// - `precursor_peak`: The peak that we are treating as the selected target ion
    /// - `isolation_window`: The selected range of m/z for the `precursor_peak`, if any.
    /// - `relative_intensity_threshold`: The minimum percentage of the `precursor_peak`'s intensity
    ///   needed for a coisolating ion to be be reported.
    /// - `ignore_singly_charged`: Whether or not to skip singly charged ions near the `precursor_peak`
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

    /// Estimate the purity of an ion selection as a function of the ratio of the selected ion's
    /// total intensity to the total intensity of all co-isolating isotopic peak.
    ///
    /// # Arguments
    /// `peaks`: The original raw m/z peak list.
    /// `precursor_peak`: The deconvolved selected ion's merged peak.
    /// `isolation_window`: The selected range of m/z for the `precursor_peak`, if any.
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
