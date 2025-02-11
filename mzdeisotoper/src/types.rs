use mzdata::{params::ParamDescribed, prelude::IonMobilityFrameLike, spectrum::{FeatureDataLevel, IonMobilityFrameDescription, IonMobilityFrameGroup, MultiLayerIonMobilityFrame, MultiLayerSpectrum, RefFeatureDataLevel, SpectrumGroup}, RawSpectrum};
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



/// Some formats write ion mobility frames by flattening them into [`RawSpectrum`],
/// so we can save some work on the writer task's end by having the worker tasks pre-
/// flatten and compress them.
///
/// This ADT shouldn't be used for anything but reading metadata and pushing the data
/// through to the writer task.
#[derive(Debug)]
pub(crate) enum FrameResult {
    Flattened(RawSpectrum, IonMobilityFrameDescription),
    Frame(FrameType),
}


pub(crate) type FrameResultGroup = IonMobilityFrameGroup<CFeature, DFeature, FrameResult>;

impl Default for FrameResult {
    fn default() -> Self {
        Self::Frame(Default::default())
    }
}

macro_rules! frdisp {
    ($d:ident, $r:ident, $e:expr) => {
        match $d {
            FrameResult::Flattened(_, $r) => $e,
            FrameResult::Frame($r) => $e,
        }
    };
}

impl ParamDescribed for FrameResult {
    fn params(&self) -> &[mzdata::Param] {
        frdisp!(self, x, x.params())
    }

    fn params_mut(&mut self) -> &mut mzdata::ParamList {
        frdisp!(self, x, x.params_mut())
    }
}

/// Implement the minimum information needed for carrying [`FrameResult`] through
/// an [`IonMobilityFrameGroup`]
impl IonMobilityFrameLike<CFeature, DFeature> for FrameResult {
    fn description(&self) -> &mzdata::spectrum::IonMobilityFrameDescription {
        match self {
            FrameResult::Flattened(_, desc) => desc,
            FrameResult::Frame(frame) => frame.description(),
        }
    }

    fn description_mut(&mut self) -> &mut mzdata::spectrum::IonMobilityFrameDescription {
        match self {
            FrameResult::Flattened(_, desc) => desc,
            FrameResult::Frame(frame) => frame.description_mut(),
        }
    }

    fn raw_arrays(&'_ self) -> Option<&'_ mzdata::spectrum::bindata::BinaryArrayMap3D> {
        None
    }

    fn features(&self) -> RefFeatureDataLevel<CFeature, DFeature> {
        RefFeatureDataLevel::Missing
    }

    fn into_features_and_parts(
        self,
    ) -> (
        FeatureDataLevel<CFeature, DFeature>,
        mzdata::spectrum::IonMobilityFrameDescription,
    ) {
        match self {
            FrameResult::Flattened(_, desc) => (FeatureDataLevel::Missing, desc),
            FrameResult::Frame(frame) => frame.into_features_and_parts(),
        }
    }
}
