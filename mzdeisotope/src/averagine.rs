use mzdeisotope::isotopic_model::{
    IsotopicModel, IsotopicModels, IsotopicPatternGenerator, PROTON,
};
use std::env;

fn main() {
    let mut args = env::args().skip(2);
    let mut model: IsotopicModel = IsotopicModels::Peptide.into();
    let mz = args
        .next()
        .inspect(|s| eprintln!("m/z: {s}"))
        .expect("Expected a floating point m/z")
        .parse::<f64>()
        .unwrap();
    let charge = args
        .next()
        .inspect(|s| eprintln!("z: {s}"))
        .expect("Expected an integer charge")
        .parse::<i32>()
        .unwrap();
    let s = model.scale(mz, charge, PROTON);
    println!("{}", s.to_string());
    let tid = model
        .isotopic_cluster(mz, charge, PROTON, 0.95, 0.001)
        .scale_by(100.0);
    for peak in &tid {
        println!(
            "{:.3}\t{:.5}\t{}",
            peak.mz(),
            peak.intensity(),
            peak.charge()
        );
    }
}
