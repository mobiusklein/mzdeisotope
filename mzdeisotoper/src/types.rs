use mzdata::spectrum::{utils::Collator, MultiLayerSpectrum, SpectrumGroup};
use mzpeaks::CentroidPeak;

use mzdeisotope::solution::DeconvolvedSolutionPeak;

use crate::selection_targets::TargetTrackingSpectrumGroup;

pub(crate) type CPeak = CentroidPeak;
pub(crate) type DPeak = DeconvolvedSolutionPeak;
pub(crate) type SpectrumType = MultiLayerSpectrum<CPeak, DPeak>;
pub(crate) type SpectrumGroupType =
    TargetTrackingSpectrumGroup<CPeak, DPeak, SpectrumGroup<CPeak, DPeak, SpectrumType>>;
pub(crate) type SpectrumCollator = Collator<SpectrumType>;
pub(crate) const BUFFER_SIZE: usize = 10_000;
pub(crate) const PEAK_COUNT_THRESHOLD_WARNING: usize = 10_000;