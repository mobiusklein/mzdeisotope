//! A basic demonstration of how to use [`mzdeisotope::deconvolute_peaks`]

use std::{collections::BTreeMap, io};

use mzdata::prelude::*;
use mzdeisotope::{
    self,
    isotopic_model::{CachingIsotopicModel, IsotopicModels, IsotopicPatternParams, PROTON},
    scorer,
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
    let mut msn_isotopic_model = CachingIsotopicModel::from(IsotopicModels::Peptide);

    // Populate the isotopic pattern cache ahead of time for reproducibility between spectra.
    // These parameters are chosen to work well with MSn spectra with small, shallow isotopic distributions.
    // MS1 spectra tend to be more complex and follow different parameters.
    let isotopic_params = IsotopicPatternParams::new(0.8, 0.001, None, PROTON);
    msn_isotopic_model.populate_cache_params(50.0, 3000.0, 1, 8, isotopic_params);

    // Do the deconvolution. This involves setting a few more parameters.
    // It also takes an owned peak list because it needs to make arbitrary modifications to the
    // data, mutating the intensity or potentially deleting peaks entirely.
    let deconv_peaks = mzdeisotope::deconvolute_peaks(
        peaks.clone(),
        msn_isotopic_model,
        Tolerance::PPM(20.0),
        (1, prec_z),
        // Use the default settings from the MS-Deconv paper
        scorer::MSDeconvScorer::default(),
        // Require a goodness-of-fit score of at least 10, with higher is better
        scorer::MaximizingFitFilter::new(10.0),
        // Tolerate at most one missing isotopic peak
        1,
        isotopic_params,
        // Do use Hoopman's QuickCharge algorithm to prune tested charge states
        // without doing a full fitting
        true,
    )
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
