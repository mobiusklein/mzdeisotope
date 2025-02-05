#[allow(unused)]
use std::{fs, io, io::prelude::*};

use mzdata::{
    io::mzml::MzMLWriterType,
    params::Unit,
    prelude::*,
    spectrum::{
        bindata::ArrayRetrievalError, ArrayType, BinaryArrayMap, BinaryDataArrayType, DataArray,
        MultiLayerSpectrum,
    },
    utils::{mass_charge_ratio, neutral_mass},
    Param,
};

use mzpeaks::{
    peak_set::PeakSetVec, CentroidPeak, DeconvolutedPeak, IonMobility, Mass, Tolerance, MZ,
};
use mzsignal::feature_statistics::FeatureTransform;

use mzdeisotope::isotopic_model::{CachingIsotopicModel, IsotopicModels};
use mzdeisotope::scorer::{MaximizingFitFilter, PenalizedMSDeconvScorer};
use mzdeisotope_map::{
    solution::DeconvolvedSolutionFeature, FeatureProcessor, FeatureSearchParams,
};

#[derive(Debug, Default, Clone, PartialEq, PartialOrd)]
struct IonMobilityAwareDeconvolutedPeak {
    pub neutral_mass: f64,
    pub ion_mobility: f64,
    pub charge: i32,
    pub intensity: f32,
    pub index: u32,
}

impl IonMobilityAwareDeconvolutedPeak {
    fn new(neutral_mass: f64, ion_mobility: f64, charge: i32, intensity: f32, index: u32) -> Self {
        Self {
            neutral_mass,
            ion_mobility,
            charge,
            intensity,
            index,
        }
    }
}

impl CoordinateLike<Mass> for IonMobilityAwareDeconvolutedPeak {
    fn coordinate(&self) -> f64 {
        self.neutral_mass
    }
}

impl CoordinateLike<MZ> for IonMobilityAwareDeconvolutedPeak {
    fn coordinate(&self) -> f64 {
        mass_charge_ratio(self.neutral_mass, self.charge)
    }
}

impl CoordinateLike<IonMobility> for IonMobilityAwareDeconvolutedPeak {
    fn coordinate(&self) -> f64 {
        self.ion_mobility
    }
}

impl From<IonMobilityAwareDeconvolutedPeak> for DeconvolutedPeak {
    fn from(value: IonMobilityAwareDeconvolutedPeak) -> Self {
        Self::new(
            value.neutral_mass,
            value.intensity,
            value.charge,
            value.index,
        )
    }
}

impl IntensityMeasurement for IonMobilityAwareDeconvolutedPeak {
    fn intensity(&self) -> f32 {
        self.intensity
    }
}

impl KnownCharge for IonMobilityAwareDeconvolutedPeak {
    fn charge(&self) -> i32 {
        self.charge
    }
}

impl IndexedCoordinate<Mass> for IonMobilityAwareDeconvolutedPeak {
    fn get_index(&self) -> mzpeaks::IndexType {
        self.index
    }

    fn set_index(&mut self, index: mzpeaks::IndexType) {
        self.index = index;
    }
}

impl BuildArrayMapFrom for IonMobilityAwareDeconvolutedPeak {
    fn as_arrays(source: &[Self]) -> BinaryArrayMap {
        let mut mz_array =
            DataArray::from_name_and_type(&ArrayType::MZArray, BinaryDataArrayType::Float64);
        let tmp: Vec<_> = source.iter().map(|p| p.mz()).collect();
        mz_array.extend(&tmp).unwrap();

        let mut int_array =
            DataArray::from_name_and_type(&ArrayType::IntensityArray, BinaryDataArrayType::Float32);
        let tmp: Vec<_> = source.iter().map(|p| p.intensity()).collect();
        int_array.extend(&tmp).unwrap();

        let mut z_array =
            DataArray::from_name_and_type(&ArrayType::ChargeArray, BinaryDataArrayType::Int32);
        let tmp: Vec<_> = source.iter().map(|p| p.charge()).collect();
        z_array.extend(&tmp).unwrap();

        let mut im_array = DataArray::from_name_and_type(
            &ArrayType::DeconvolutedIonMobilityArray,
            BinaryDataArrayType::Float64,
        );
        let tmp: Vec<_> = source.iter().map(|p| p.ion_mobility()).collect();
        im_array.extend(&tmp).unwrap();
        im_array.unit = Unit::VoltSecondPerSquareCentimeter;

        let mut map = BinaryArrayMap::new();
        map.add(mz_array);
        map.add(int_array);
        map.add(z_array);
        map.add(im_array);

        map
    }

    fn arrays_included(&self) -> Option<Vec<ArrayType>> {
        Some(vec![
            ArrayType::MZArray,
            ArrayType::IntensityArray,
            ArrayType::ChargeArray,
            ArrayType::DeconvolutedIonMobilityArray,
        ])
    }
}

impl BuildFromArrayMap for IonMobilityAwareDeconvolutedPeak {
    fn try_from_arrays(arrays: &BinaryArrayMap) -> Result<Vec<Self>, ArrayRetrievalError> {
        let mzs = arrays.mzs()?;
        let intens = arrays.intensities()?;
        let charges = arrays.charges()?;
        let ims = arrays
            .get(&ArrayType::DeconvolutedIonMobilityArray)
            .ok_or(ArrayRetrievalError::NotFound(
                ArrayType::DeconvolutedIonMobilityArray,
            ))?
            .to_f64()?;

        let mut peaks = Vec::new();
        for (mz, (intens, (z, im))) in mzs.iter().zip(
            intens
                .iter()
                .zip(charges.iter().zip(ims.iter())),
        ) {
            let mass = neutral_mass(*mz, *z);
            peaks.push(IonMobilityAwareDeconvolutedPeak::new(
                mass, *im, *z, *intens, 0,
            ));
        }

        Ok(peaks)
    }

    fn arrays_required() -> Option<Vec<ArrayType>> {
        Some(vec![
            ArrayType::MZArray,
            ArrayType::IntensityArray,
            ArrayType::ChargeArray,
            ArrayType::DeconvolutedIonMobilityArray,
        ])
    }
}

fn main() -> io::Result<()> {
    let sid = "merged=42926 frame=9728 scanStart=1 scanEnd=705";

    let fh = fs::File::create("ims_example.mzML")?;
    let mut writer: MzMLWriterType<fs::File, CentroidPeak, IonMobilityAwareDeconvolutedPeak> =
        MzMLWriterType::new(fh);

    #[allow(unexpected_cfgs)]
    let mut frame = mzdata::mz_read!("test/data/20200204_BU_8B8egg_1ug_uL_7charges_60_min_Slot2-11_1_244.mzML.gz".as_ref(), reader => {
        writer.copy_metadata_from(&reader);
        let mut reader = mzdata::io::Generic3DIonMobilityFrameSource::new(reader);
        let frame: mzdata::spectrum::MultiLayerIonMobilityFrame<_, DeconvolvedSolutionFeature<IonMobility>> = reader.get_frame_by_id(sid).unwrap();

        frame
    })?;

    frame.add_param(Param::builder().name("ion mobility profile frame").build());
    let scan = frame.description_mut().acquisition.last_scan_mut().unwrap();
    scan.remove_param(
        scan.iter_params()
            .position(|p| p.name() == "inverse reduced ion mobility")
            .unwrap(),
    );

    let i = frame.iter_params().position(|p| p.name() == "ion mobility lower limit").unwrap();
    frame.params_mut()[i].name = "lowest observed ion mobility".to_string();
    let i = frame.iter_params().position(|p| p.name() == "ion mobility upper limit").unwrap();
    frame.params_mut()[i].name = "highest observed ion mobility".to_string();

    writer.set_spectrum_count(4);
    writer.write_frame(&frame)?;

    frame.extract_features_simple(Tolerance::PPM(15.0), 2, 0.01, None)?;
    frame.description_mut().id += ".1";
    frame.params_mut().pop().unwrap();
    frame.add_param(Param::builder().name("ion mobility feature frame").build());
    writer.write_frame(&frame)?;

    frame
        .features
        .as_mut()
        .map(|fmap| {
            let mut alt = Default::default();
            std::mem::swap(&mut alt, fmap);
            *fmap = alt
                .into_iter()
                .filter(|f| f.len() > 1)
                .map(|mut f| {
                    f.smooth(1);
                    f
                })
                .collect();
            fmap
        })
        .unwrap();

    let mut deconv = FeatureProcessor::new(
        frame.features.clone().unwrap(),
        CachingIsotopicModel::from(IsotopicModels::Glycopeptide),
        PenalizedMSDeconvScorer::new(0.04, 2.0),
        MaximizingFitFilter::new(5.0),
        2,
        0.02,
        5.0,
        true,
    );

    let params = FeatureSearchParams {
        truncate_after: 0.95,
        ignore_below: 0.05,
        max_missed_peaks: 2,
        threshold_scale: 0.3,
        detection_threshold: 0.1,
    };

    let deconv_map = deconv
        .deconvolve(Tolerance::PPM(15.0), (1, 8), 1, 1, &params, 1e-3, 10)
        .unwrap();

    frame.deconvoluted_features = Some(deconv_map);
    frame.description_mut().id = frame.description_mut().id.replace(".1", ".2");
    writer.write_frame(&frame)?;

    let peaks: PeakSetVec<IonMobilityAwareDeconvolutedPeak, Mass> = frame
        .deconvoluted_features
        .as_ref()
        .unwrap()
        .iter()
        .map(|f| {
            IonMobilityAwareDeconvolutedPeak::new(
                f.neutral_mass(),
                f.apex_time().unwrap(),
                f.charge(),
                f.intensity(),
                0,
            )
        })
        .collect();

    let descr = frame.description().clone();
    let mut spec: MultiLayerSpectrum<CentroidPeak, IonMobilityAwareDeconvolutedPeak> =
        MultiLayerSpectrum::new(descr.into(), None, None, Some(peaks));
    spec.description_mut().id = spec.description_mut().id.replace(".2", ".3");
    spec.params_mut().pop().unwrap();
    spec.add_param(Param::builder().name("ion mobility centroid frame").build());
    writer.write_spectrum(&spec)?;
    Ok(())
}
