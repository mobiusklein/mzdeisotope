mod feature_fit;
mod processor;
mod iter;
mod dependency_graph;
mod traits;
mod api;
mod fmap;

pub mod solution;
pub mod prelude;

pub use api::{FeatureDeconvolutionEngine, deconvolute_features};
pub use processor::{FeatureProcessor, EnvelopeConformer};
pub use traits::{FeatureSearchParams, DeconvolutionError};
pub use iter::FeatureSetIter;
pub use feature_fit::{FeatureSetFit, MapCoordinate};