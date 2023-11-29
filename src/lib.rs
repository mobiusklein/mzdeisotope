pub mod charge;
pub mod deconv_traits;
pub mod isotopic_fit;
pub mod isotopic_model;
pub mod peak_graph;
pub mod peaks;
pub mod scorer;
pub mod solution;

pub mod deconvoluter;
pub mod interval;

#[cfg(test)]
mod test {
    use flate2::bufread::GzDecoder;
    use mzdata::MzMLReader;
    use mzpeaks::prelude::*;
    use std::fs;
    use std::io;

    use crate::isotopic_model::PROTON;
    use crate::isotopic_model::{IsotopicModel, IsotopicModels, IsotopicPatternGenerator};
    use crate::peaks::WorkingPeakSet;
    use crate::scorer::MSDeconvScorer;

    #[test]
    fn test_file() -> io::Result<()> {
        let decoder = GzDecoder::new(io::BufReader::new(fs::File::open(
            "./tests/data/20150710_3um_AGP_001_29_30.mzML.gz",
        )?));
        let mut reader = MzMLReader::new(decoder);
        let scan = reader.next().unwrap();
        let tol = Tolerance::PPM(10.0);
        let centroided = scan.into_centroid().unwrap();
        let peaks_at = centroided.peaks.all_peaks_for(1161.0092, tol);
        let best_peak = peaks_at
            .iter()
            .max_by(|a, b| a.intensity().partial_cmp(&b.intensity).unwrap())
            .unwrap()
            .clone();
        assert!(centroided.peaks.has_peak(best_peak.mz(), tol).is_some());
        let mut peaks = WorkingPeakSet::new(centroided.peaks);
        let (_key, placeholder) = peaks.has_peak(best_peak.mz(), tol);
        assert!(!placeholder);

        let mut isotopic_model: IsotopicModel = IsotopicModels::Glycopeptide.into();
        let tid = isotopic_model.isotopic_cluster(best_peak.mz(), 4, PROTON, 0.95, 0.01);
        let (eid_keys, missed) = peaks.match_theoretical(&tid, tol);
        assert_eq!(eid_keys.len(), tid.len());
        assert_eq!(missed, 0);

        let scorer = MSDeconvScorer::new(0.02);
        let eid = peaks.collect_for(&eid_keys);
        let max_int: f32 = eid.iter().map(|p| p.intensity()).sum();
        let tid = tid.scale_by(max_int as f64);
        let score = scorer.score(&eid, &tid);
        assert!((score - 2735.188267035904).abs() < 1e-3);
        Ok(())
    }
}
