mod feature_fit;
mod processor;
mod iter;
mod dependency_graph;
mod solution;

pub use processor::{FeatureProcessor, EnvelopeConformer, FeatureSearchParams, DeconvolutionError};
pub use iter::FeatureSetIter;
pub use feature_fit::{FeatureSetFit, MapCoordinate};
pub use solution::DeconvolvedSolutionFeature;