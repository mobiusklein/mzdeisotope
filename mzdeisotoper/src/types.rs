use mzpeaks::CentroidPeak;
use mzdata::spectrum::{MultiLayerSpectrum, SpectrumGroup, utils::Collator};

use mzdeisotope::solution::DeconvolvedSolutionPeak;

use crate::selection_targets::TargetTrackingSpectrumGroup;

pub(crate) type CPeak = CentroidPeak;
pub(crate) type DPeak = DeconvolvedSolutionPeak;
pub(crate) type SpectrumType = MultiLayerSpectrum<CPeak, DPeak>;
pub(crate) type SpectrumGroupType =
    TargetTrackingSpectrumGroup<CPeak, DPeak, SpectrumGroup<CPeak, DPeak, SpectrumType>>;
pub(crate) type SpectrumGroupCollator = Collator<SpectrumGroupType>;