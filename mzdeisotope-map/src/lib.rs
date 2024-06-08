mod feature_fit;
mod processor;
mod iter;
mod dependency_graph;
mod traits;
pub mod solution;
pub mod prelude;

pub use processor::{FeatureProcessor, EnvelopeConformer};
pub use traits::{FeatureSearchParams, DeconvolutionError};
pub use iter::FeatureSetIter;
pub use feature_fit::{FeatureSetFit, MapCoordinate};