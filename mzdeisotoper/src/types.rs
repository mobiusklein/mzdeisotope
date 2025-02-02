use mzdata::spectrum::{MultiLayerSpectrum, SpectrumGroup, MultiLayerIonMobilityFrame, IonMobilityFrameGroup};
use mzpeaks::{CentroidPeak, feature::Feature, MZ, IonMobility};

use mzdeisotope::solution::DeconvolvedSolutionPeak;
use mzdeisotope_map::solution::DeconvolvedSolutionFeature;

use crate::selection_targets::TargetTrackingSpectrumGroup;

pub(crate) const BUFFER_SIZE: usize = 10_000;
pub(crate) const PEAK_COUNT_THRESHOLD_WARNING: usize = 10_000;

pub(crate) type CPeak = CentroidPeak;
pub(crate) type DPeak = DeconvolvedSolutionPeak;
pub(crate) type SpectrumType = MultiLayerSpectrum<CPeak, DPeak>;
pub(crate) type SpectrumGroupType =
    TargetTrackingSpectrumGroup<CPeak, DPeak, SpectrumGroup<CPeak, DPeak, SpectrumType>>;
// pub(crate) type SpectrumCollator = Collator<SpectrumType>;

pub(crate) type CFeature = Feature<MZ, IonMobility>;
pub(crate) type DFeature = DeconvolvedSolutionFeature<IonMobility>;
pub(crate) type FrameType = MultiLayerIonMobilityFrame<CFeature, DFeature>;
pub(crate) type FrameGroupType = IonMobilityFrameGroup<CFeature, DFeature, FrameType>;
// pub(crate) type FrameCollator = Collator<FrameType>;
