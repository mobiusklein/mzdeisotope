use std::{cmp, mem};

use itertools::multizip;

use mzdata::spectrum::{BinaryArrayMap, BinaryDataArrayType, DataArray};
use mzpeaks;
use mzpeaks::peak::MZPoint;
use mzpeaks::prelude::*;
use mzpeaks::{CoordinateLike, IntensityMeasurement, KnownCharge, Mass, MZ};

use mzdata::spectrum::bindata::{
    ArrayRetrievalError, ArrayType, BinaryCompressionType, BuildArrayMapFrom, BuildFromArrayMap,
    ByteArrayView,
};

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

impl CoordinateLike<MZ> for DeconvolvedSolutionPeak {
    fn coordinate(&self) -> f64 {
        self.mz()
    }
}

const DECONVOLUTION_SCORE_ARRAY_NAME: &str = "deconvolution score array";
const ISOTOPIC_ENVELOPE_ARRAY_NAME: &str = "isotopic envelopes array";

impl BuildFromArrayMap for DeconvolvedSolutionPeak {
    fn try_from_arrays(arrays: &BinaryArrayMap) -> Result<Vec<Self>, ArrayRetrievalError> {
        let mz_array = arrays.mzs()?;

        let intensity_array = arrays.intensities()?;

        let charge_array = arrays.charges()?;

        let score_array = match arrays.get(&ArrayType::NonStandardDataArray {
            name: Box::new(DECONVOLUTION_SCORE_ARRAY_NAME.into()),
        }) {
            Some(a) => a.to_f32()?,
            None => {
                return Err(ArrayRetrievalError::NotFound(
                    ArrayType::NonStandardDataArray {
                        name: Box::new(DECONVOLUTION_SCORE_ARRAY_NAME.to_string()),
                    },
                ))
            }
        };

        let isotopic_envelopes_array = match arrays.get(&ArrayType::NonStandardDataArray {
            name: Box::new(ISOTOPIC_ENVELOPE_ARRAY_NAME.into()),
        }) {
            Some(a) => a.to_f32()?,
            None => {
                return Err(ArrayRetrievalError::NotFound(
                    ArrayType::NonStandardDataArray {
                        name: Box::new(ISOTOPIC_ENVELOPE_ARRAY_NAME.to_string()),
                    },
                ))
            }
        };

        let mut envelopes_acc = Vec::with_capacity(mz_array.len());
        isotopic_envelopes_array.windows(2).step_by(2).fold(
            Envelope::new(),
            |mut current_envelope, point| {
                if point[0] == 0.0 && point[1] == 0.0 {
                    envelopes_acc.push(current_envelope);
                    Envelope::new()
                } else {
                    current_envelope.push(MZPoint::new(point[0] as f64, point[1]));
                    current_envelope
                }
            },
        );

        let mut peaks = Vec::with_capacity(mz_array.len());

        peaks.extend(
            multizip((
                mz_array.iter(),
                intensity_array.iter(),
                charge_array.iter(),
                score_array.iter(),
                envelopes_acc.into_iter(),
            ))
            .map(|(neutral_mass, intensity, charge, score, envelope)| {
                DeconvolvedSolutionPeak::new(
                    *neutral_mass,
                    *intensity,
                    *charge,
                    0,
                    *score,
                    Box::new(envelope),
                )
            }),
        );
        Ok(peaks)
    }
}

impl BuildArrayMapFrom for DeconvolvedSolutionPeak {
    fn as_arrays(source: &[Self]) -> BinaryArrayMap {
        let mut arrays = BinaryArrayMap::new();

        let mut mz_array = DataArray::from_name_type_size(
            &ArrayType::MZArray,
            BinaryDataArrayType::Float64,
            source.len() * BinaryDataArrayType::Float64.size_of(),
        );

        let mut intensity_array = DataArray::from_name_type_size(
            &ArrayType::IntensityArray,
            BinaryDataArrayType::Float32,
            source.len() * BinaryDataArrayType::Float32.size_of(),
        );

        let mut charge_array = DataArray::from_name_type_size(
            &ArrayType::ChargeArray,
            BinaryDataArrayType::Int32,
            source.len() * BinaryDataArrayType::Int32.size_of(),
        );

        let mut score_array = DataArray::from_name_type_size(
            &ArrayType::NonStandardDataArray {
                name: Box::new(DECONVOLUTION_SCORE_ARRAY_NAME.into()),
            },
            BinaryDataArrayType::Float32,
            source.len() * BinaryDataArrayType::Float32.size_of(),
        );

        let mut envelope_array = DataArray::from_name_type_size(
            &ArrayType::NonStandardDataArray {
                name: Box::new(ISOTOPIC_ENVELOPE_ARRAY_NAME.into()),
            },
            BinaryDataArrayType::Float32,
            source.iter().map(|p| p.envelope.len() + 2).sum::<usize>()
                * BinaryDataArrayType::Float32.size_of(),
        );

        mz_array.compression = BinaryCompressionType::Decoded;
        intensity_array.compression = BinaryCompressionType::Decoded;
        charge_array.compression = BinaryCompressionType::Decoded;
        score_array.compression = BinaryCompressionType::Decoded;
        envelope_array.compression = BinaryCompressionType::Decoded;

        for p in source.iter() {
            let mz: f64 = p.mz();
            let inten: f32 = p.intensity();
            let charge = p.charge();

            let raw_bytes: [u8; mem::size_of::<f64>()] = mz.to_le_bytes();
            mz_array.data.extend(raw_bytes);

            let raw_bytes: [u8; mem::size_of::<f32>()] = inten.to_le_bytes();
            intensity_array.data.extend(raw_bytes);

            let raw_bytes: [u8; mem::size_of::<i32>()] = charge.to_le_bytes();
            charge_array.data.extend(raw_bytes);

            let raw_bytes: [u8; mem::size_of::<f32>()] = p.score.to_le_bytes();
            score_array.data.extend(raw_bytes);

            p.envelope.iter().for_each(|pt| {
                let raw_bytes: [u8; mem::size_of::<f32>()] = (pt.mz as f32).to_le_bytes();
                envelope_array.data.extend(raw_bytes);
                let raw_bytes: [u8; mem::size_of::<f32>()] = pt.intensity.to_le_bytes();
                envelope_array.data.extend(raw_bytes);
            });

            envelope_array.data.extend(0.0f32.to_le_bytes());
            envelope_array.data.extend(0.0f32.to_le_bytes());
        }

        arrays.add(mz_array);
        arrays.add(intensity_array);
        arrays.add(charge_array);
        arrays.add(score_array);
        arrays.add(envelope_array);
        arrays
    }
}
