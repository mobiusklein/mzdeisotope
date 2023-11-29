use std::cmp;

use mzpeaks;
use mzpeaks::peak::MZPoint;
use mzpeaks::prelude::*;
use mzpeaks::{IntensityMeasurement, KnownCharge};

use crate::scorer::ScoreType;

pub type Envelope = Vec<MZPoint>;

#[derive(Debug, Default, Clone)]
pub struct DeconvolvedSolutionPeak {
    pub neutral_mass: f64,
    pub intensity: f32,
    pub charge: i32,
    pub index: u32,
    pub score: ScoreType,
    pub envelope: Box<Envelope>,
}

impl DeconvolvedSolutionPeak {
    pub fn new(
        neutral_mass: f64,
        intensity: f32,
        charge: i32,
        index: u32,
        score: ScoreType,
        envelope: Box<Envelope>,
    ) -> Self {
        Self {
            neutral_mass,
            intensity,
            charge,
            index,
            score,
            envelope,
        }
    }

    pub fn mz(&self) -> f64 {
        let charge_carrier: f64 = 1.007276;
        let charge = self.charge as f64;
        (self.neutral_mass + charge_carrier * charge) / charge
    }
}

mzpeaks::implement_deconvoluted_centroidlike!(DeconvolvedSolutionPeak, true);
