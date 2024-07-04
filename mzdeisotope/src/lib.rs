//! A library for deisotoping and charge state deconvolution of complex mass spectra.
//!
//!
pub(crate) mod api;
pub mod charge;
pub mod deconv_traits;
pub mod isolation;
pub mod isotopic_fit;
pub mod isotopic_model;
pub mod peak_graph;
pub mod peaks;
pub mod scorer;
pub mod solution;

pub mod deconvoluter;
pub mod multi_model_deconvoluters;

#[doc(inline)]
pub use solution::DeconvolvedSolutionPeak;
pub use api::{
    deconvolute_peaks, deconvolute_peaks_with_targets, DeconvolutionEngine, IsotopicModelLike,
    PeaksAndTargets,
};

