//! A basic demonstration of how to use [`mzdeisotope::DeconvolutionEngine`].
//!
//! While functionally equivalent to [`mzdeisotope::deconvolute_peaks`], the
//! [`DeconvolutionEngine`] type encapsulates all of the
//! parts that are not directly data dependent. Internally, [`deconvolute_peaks`]
//! is just taking all of the arguments, creating a [`DeconvolutionEngine`], and
//! uses it once and throws it away after computing the result.

use std::{collections::BTreeMap, io};

use mzdata::prelude::*;
use mzdeisotope::{
    isotopic_model::{CachingIsotopicModel, IsotopicModels, IsotopicPatternParams, PROTON},
    scorer, DeconvolutionEngine,
};

fn main() -> io::Result<()> {
    // Set up for example, read an MSn spectrum from a file.
    let mut reader = mzdata::MZReader::open_path("test/data/batching_test.mzML")?;
    let mut spec = reader.get_spectrum_by_index(0).unwrap();

    // Find the precursor ion charge state. Derived product ions cannot have a charge higher than this!
    let prec_z = spec.precursor().as_ref().unwrap().ion().charge().unwrap();

    // Pick peaks from the spectrum so there will be centroids.
    spec.pick_peaks(1.0).unwrap();
    let peaks = spec.peaks.as_ref().unwrap();

    // Create an isotopic pattern model that follows the peptide averagine (Senko et al.)
    // that we will use for processing MS2 spectra
    let msn_isotopic_model = CachingIsotopicModel::from(IsotopicModels::Peptide);

    let isotopic_params = IsotopicPatternParams::new(0.8, 0.001, None, PROTON);
    let mut engine = DeconvolutionEngine::new(
        isotopic_params,
        msn_isotopic_model,
        scorer::MSDeconvScorer::default(),
        scorer::MaximizingFitFilter::new(10.0),
        true,
    );

    // Populate the isotopic pattern cache ahead of time for reproducibility between spectra.
    // These parameters are chosen to work well with MSn spectra with small, shallow isotopic distributions.
    // MS1 spectra tend to be more complex and follow different parameters.
    engine.populate_isotopic_model_cache(50.0, 3000.0, 1, 8);

    // Do the deconvolution. It takes an owned peak list because it needs to make arbitrary modifications to the
    // data, mutating the intensity or potentially deleting peaks entirely.
    let deconv_peaks = engine
        .deconvolute_peaks(peaks.clone(), Tolerance::PPM(20.0), (1, prec_z), 1)
        .unwrap(); // Assume it's safe to unwrap here.

    // Let's count some charge states to prove something happened
    let mut table: BTreeMap<i32, usize> = BTreeMap::new();
    for peak in deconv_peaks.iter() {
        *table.entry(peak.charge()).or_default() += 1;
    }

    for (z, count) in table.iter() {
        eprintln!("{z} => {count}");
    }

    let n_raw = peaks.len();
    let n_deconv = deconv_peaks.len();
    let n_envelope = deconv_peaks
        .iter()
        .map(|p| p.envelope.iter().filter(|i| i.intensity > 1.0).count())
        .sum::<usize>();
    eprintln!("{n_raw} raw centroids");
    eprintln!("{n_deconv} deconvolved centroids built over {n_envelope} raw centroids");

    Ok(())
}
