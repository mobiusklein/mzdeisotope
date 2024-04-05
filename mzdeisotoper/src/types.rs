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
