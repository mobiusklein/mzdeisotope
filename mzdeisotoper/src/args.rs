use std::fmt::Display;

use clap::ValueEnum;

use mzdeisotope::{
    api::DeconvolutionEngine,
    charge::ChargeRange,
    isotopic_model::{IsotopicModel, IsotopicModels, IsotopicPatternParams, PROTON},
    scorer::{
        IsotopicFitFilter, IsotopicPatternScorer, MSDeconvScorer, MaximizingFitFilter,
        PenalizedMSDeconvScorer,
    },
};
use mzpeaks::CentroidPeak;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum PrecursorProcessing {
    #[default]
    /// Process the entire MS1 mass range and all MSn spectra
    Full,
    /// Process only the MS1 regions that are selected for MSn and all MSn spectra
    SelectedPrecursors,
    /// Process only MSn spectra without examining MS1 spectra
    TandemOnly,
    /// Process only the MS1 spectra without examining MSn spectra
    MS1Only,
}

impl Display for PrecursorProcessing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum ArgIsotopicModels {
    Peptide,
    Glycan,
    Glycopeptide,
    PermethylatedGlycan,
    Heparin,
    HeparanSulfate,
}

impl Into<IsotopicModels> for ArgIsotopicModels {
    fn into(self) -> IsotopicModels {
        match self {
            ArgIsotopicModels::Peptide => IsotopicModels::Peptide,
            ArgIsotopicModels::Glycan => IsotopicModels::Glycan,
            ArgIsotopicModels::Glycopeptide => IsotopicModels::Glycopeptide,
            ArgIsotopicModels::PermethylatedGlycan => IsotopicModels::PermethylatedGlycan,
            ArgIsotopicModels::Heparin => IsotopicModels::Heparin,
            ArgIsotopicModels::HeparanSulfate => IsotopicModels::HeparanSulfate,
        }
    }
}

impl Into<IsotopicModel<'static>> for ArgIsotopicModels {
    fn into(self) -> IsotopicModel<'static> {
        let m: IsotopicModels = self.into();
        m.into()
    }
}

impl Display for ArgIsotopicModels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SignalParams {
    pub ms1_averaging: usize,
    pub mz_range: (f64, f64),
    pub interpolation_dx: f64,
    pub ms1_denoising: f32,
}

pub struct DeconvolutionBuilderParams<'a, S: IsotopicPatternScorer, F: IsotopicFitFilter> {
    pub scorer: S,
    pub isotopic_model: IsotopicModel<'a>,
    pub fit_filter: F,
    pub isotopic_params: IsotopicPatternParams,
    pub charge_range: ChargeRange,
    pub mz_range: (f64, f64),
    pub max_missed_peaks: u16,
}

impl<'a, S: IsotopicPatternScorer, F: IsotopicFitFilter> DeconvolutionBuilderParams<'a, S, F> {
    pub fn new(
        scorer: S,
        isotopic_model: IsotopicModel<'a>,
        fit_filter: F,
        isotopic_params: IsotopicPatternParams,
        charge_range: ChargeRange,
        mz_range: (f64, f64),
        max_missed_peaks: u16,
    ) -> Self {
        Self {
            scorer,
            isotopic_model,
            fit_filter,
            isotopic_params,
            charge_range,
            mz_range,
            max_missed_peaks,
        }
    }

    pub fn make_params(&self) -> DeconvolutionParams {
        DeconvolutionParams::new(self.charge_range, self.max_missed_peaks)
    }

    pub fn build_engine(self) -> DeconvolutionEngine<'a, CentroidPeak, S, F> {
        let mut engine = DeconvolutionEngine::new(
            self.isotopic_params,
            self.isotopic_model.into(),
            self.scorer,
            self.fit_filter,
            true,
        );
        engine.populate_isotopic_model_cache(
            self.mz_range.0,
            self.mz_range.1,
            self.charge_range.0,
            self.charge_range.1,
        );
        engine
    }

    pub fn make_params_and_engine(
        self,
    ) -> (
        DeconvolutionParams,
        DeconvolutionEngine<'a, CentroidPeak, S, F>,
    ) {
        let params = self.make_params();
        let engine = self.build_engine();
        (params, engine)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DeconvolutionParams {
    pub charge_range: ChargeRange,
    pub max_missed_peaks: u16,
}

impl DeconvolutionParams {
    pub fn new(charge_range: ChargeRange, max_missed_peaks: u16) -> Self {
        Self {
            charge_range,
            max_missed_peaks,
        }
    }
}

pub fn make_default_ms1_deconvolution_params(
) -> DeconvolutionBuilderParams<'static, PenalizedMSDeconvScorer, MaximizingFitFilter> {
    DeconvolutionBuilderParams::new(
        PenalizedMSDeconvScorer::new(0.02, 2.0),
        IsotopicModels::Peptide.into(),
        MaximizingFitFilter::new(20.0),
        Default::default(),
        (1, 8),
        (80.0, 2200.0),
        1,
    )
}

pub fn make_default_msn_deconvolution_params(
) -> DeconvolutionBuilderParams<'static, MSDeconvScorer, MaximizingFitFilter> {
    DeconvolutionBuilderParams::new(
        MSDeconvScorer::default(),
        IsotopicModels::Peptide.into(),
        MaximizingFitFilter::new(5.0),
        IsotopicPatternParams::new(0.8, 0.001, None, PROTON),
        (1, 8),
        (80.0, 2200.0),
        1,
    )
}

pub fn make_default_signal_processing_params() -> SignalParams {
    SignalParams {
        ms1_averaging: 1,
        mz_range: (80.0, 2200.0),
        interpolation_dx: 0.005,
        ms1_denoising: 0.0,
    }
}
