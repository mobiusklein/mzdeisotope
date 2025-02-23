use std::{fmt::Display, str::FromStr};

use clap::ValueEnum;
use mzdeisotope_map::{FeatureDeconvolutionEngine, FeatureSearchParams};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use mzdeisotope::{
    DeconvolutionEngine,
    charge::ChargeRange,
    isotopic_model::{IsotopicModel, IsotopicModels, IsotopicPatternParams, PROTON},
    scorer::{
        IsotopicFitFilter, IsotopicPatternScorer, MSDeconvScorer, MaximizingFitFilter,
        PenalizedMSDeconvScorer,
    },
};
use mzpeaks::{CentroidPeak, IonMobility, Tolerance};

use crate::types::CFeature;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default, Deserialize, Serialize)]
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
    /// Process all data as `full`, but ignore selected ion information
    DIA,
}

impl Display for PrecursorProcessing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Deserialize, Serialize)]
pub enum ArgIsotopicModels {
    Peptide,
    Glycan,
    Glycopeptide,
    PermethylatedGlycan,
    Heparin,
    HeparanSulfate,
}

impl From<ArgIsotopicModels> for IsotopicModels {
    fn from(value: ArgIsotopicModels) -> Self {
        match value {
            ArgIsotopicModels::Peptide => IsotopicModels::Peptide,
            ArgIsotopicModels::Glycan => IsotopicModels::Glycan,
            ArgIsotopicModels::Glycopeptide => IsotopicModels::Glycopeptide,
            ArgIsotopicModels::PermethylatedGlycan => IsotopicModels::PermethylatedGlycan,
            ArgIsotopicModels::Heparin => IsotopicModels::Heparin,
            ArgIsotopicModels::HeparanSulfate => IsotopicModels::HeparanSulfate,
        }
    }
}

impl From<ArgIsotopicModels> for IsotopicModel<'static> {
    fn from(value: ArgIsotopicModels) -> Self {
        let m: IsotopicModels = value.into();
        m.into()
    }
}

impl Display for ArgIsotopicModels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub struct ArgChargeRange(pub i32, pub i32);

impl Display for ArgChargeRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", self.0, self.1)
    }
}

impl Default for ArgChargeRange {
    fn default() -> Self {
        Self(1, 8)
    }
}

impl From<ArgChargeRange> for (i32, i32) {
    fn from(value: ArgChargeRange) -> Self {
        (value.0, value.1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ChargeRangeParserError {
    #[error("Error parsing integer {0}")]
    IntError(
        #[from]
        #[source]
        std::num::ParseIntError,
    ),
    #[error("Charge range cannot be empty")]
    EmptyRange,
    #[error("Charge cannot be zero")]
    ZeroCharge,
}

impl FromStr for ArgChargeRange {
    type Err = ChargeRangeParserError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let it = if s.contains(' ') {
            s.split(' ')
        } else if s.contains(':') {
            s.split(':')
        } else if s.find('-').map(|i| i > 0).unwrap_or(false)
            && s.chars().map(|t| (t == '-') as i32).sum::<i32>() == 1
        {
            s.split('-')
        } else {
            s.split(' ')
        };
        let r: Result<Vec<i32>, std::num::ParseIntError> = it.map(|t| t.parse()).collect();
        let r = r?;
        if r.is_empty() {
            Err(ChargeRangeParserError::EmptyRange)
        } else if r.len() == 1 {
            let val = r[0];
            if val == 0 {
                Err(ChargeRangeParserError::ZeroCharge)
            } else {
                Ok(Self(val.signum(), val))
            }
        } else {
            let low = *r.iter().min_by_key(|i| i.abs()).unwrap();
            let high = *r.iter().max_by_key(|i| i.abs()).unwrap();
            Ok(Self(low, high))
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct SignalParams {
    pub ms1_averaging: usize,
    pub mz_range: (f64, f64),
    pub interpolation_dx: f64,
    pub ms1_denoising: f32,
}

impl Default for SignalParams {
    fn default() -> Self {
        make_default_signal_processing_params()
    }
}

pub struct DeconvolutionBuilderParams<'a, S: IsotopicPatternScorer, F: IsotopicFitFilter> {
    pub scorer: S,
    pub isotopic_model: Vec<IsotopicModel<'a>>,
    pub fit_filter: F,
    pub isotopic_params: IsotopicPatternParams,
    pub charge_range: ChargeRange,
    pub mz_range: (f64, f64),
    pub max_missed_peaks: u16,
}

impl<'a, S: IsotopicPatternScorer, F: IsotopicFitFilter> DeconvolutionBuilderParams<'a, S, F> {
    pub fn new(
        scorer: S,
        isotopic_model: Vec<IsotopicModel<'a>>,
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

    pub fn build_feature_engine(self) -> FeatureDeconvolutionEngine<'a, IonMobility, CFeature, S, F> {
        let params = FeatureSearchParams {
            truncate_after: self.isotopic_params.truncate_after,
            ignore_below: self.isotopic_params.ignore_below,
            max_missed_peaks: self.max_missed_peaks as usize,
            ..Default::default()
        };
        let mut engine = FeatureDeconvolutionEngine::new(params, self.isotopic_model, self.scorer, self.fit_filter);
        engine.populate_isotopic_model_cache(
            self.mz_range.0,
            self.mz_range.1,
            self.charge_range.0,
            self.charge_range.1,
        );
        engine
    }

    pub fn build_engine(self) -> DeconvolutionEngine<'a, CentroidPeak, S, F> {
        let mut engine = DeconvolutionEngine::new(
            self.isotopic_params,
            self.isotopic_model,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FeatureExtractionParams {
    pub smoothing: usize,
    pub error_tolerance: Tolerance,
    pub minimum_size: usize,
    pub maximum_time_gap: f64,
}

pub fn make_default_ms1_feature_extraction_params() -> FeatureExtractionParams {
    FeatureExtractionParams {
        smoothing: 1,
        error_tolerance: Tolerance::PPM(10.0),
        minimum_size: 2,
        maximum_time_gap: 0.025,
    }
}


pub fn make_default_msn_feature_extraction_params() -> FeatureExtractionParams {
    FeatureExtractionParams {
        smoothing: 1,
        error_tolerance: Tolerance::PPM(10.0),
        minimum_size: 2,
        maximum_time_gap: 0.025,
    }
}


pub fn make_default_ms1_deconvolution_params(
) -> DeconvolutionBuilderParams<'static, PenalizedMSDeconvScorer, MaximizingFitFilter> {
    DeconvolutionBuilderParams::new(
        PenalizedMSDeconvScorer::new(0.02, 2.0),
        vec![IsotopicModels::Peptide.into()],
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
        vec![IsotopicModels::Peptide.into()],
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
        interpolation_dx: 0.002,
        ms1_denoising: 0.0,
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_charge_range_parse() -> Result<(), ChargeRangeParserError> {
        let p: ArgChargeRange = "1-8".parse()?;
        assert_eq!(p.1, 8);
        assert_eq!(p.0, 1);

        Ok(())
    }
}
