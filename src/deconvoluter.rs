use std::ops::Range;

use crate::isotopic_fit::IsotopicFit;
use crate::isotopic_model::{
    CachingIsotopicModel, IsotopicPatternGenerator, TIDScalingMethod, IsotopicPatternParams,
};
use crate::peak_graph::PeakDependenceGraph;
use crate::peak_graph::SubgraphSolverMethod;
use crate::peak_graph::cluster::DependenceCluster;
use crate::peak_graph::fit::FitRef;
use crate::peaks::{PeakKey, WorkingPeakSet};
use crate::scorer::{
    IsotopicFitFilter, IsotopicPatternScorer, MSDeconvScorer, MaximizingFitFilter,
};

use crate::deconv_traits::{ExhaustivePeakSearch, IsotopicPatternFitter, RelativePeakSearch, GraphDeconvolution};

use mzpeaks::prelude::*;
use mzpeaks::{CentroidPeak, MZPeakSetType, Tolerance};


#[derive(Debug)]
pub struct DeconvoluterType<
    C: CentroidLike + Clone + From<CentroidPeak>,
    I: IsotopicPatternGenerator,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
> {
    pub peaks: WorkingPeakSet<C>,
    pub isotopic_model: I,
    pub scorer: S,
    pub fit_filter: F,
    pub scaling_method: TIDScalingMethod,
    pub max_missed_peaks: u16,
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak>,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > RelativePeakSearch<C> for DeconvoluterType<C, I, S, F>
{
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak>,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > ExhaustivePeakSearch<C> for DeconvoluterType<C, I, S, F>
{
    fn check_isotopic_fit(&self, fit: &IsotopicFit) -> bool {
        if fit.missed_peaks > self.max_missed_peaks {
            return false;
        }
        self.fit_filter.test(fit)
    }
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak>,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > DeconvoluterType<C, I, S, F>
{
    pub fn new(
        peaks: MZPeakSetType<C>,
        isotopic_model: I,
        scorer: S,
        fit_filter: F,
        max_missed_peaks: u16,
    ) -> Self {
        Self {
            peaks: WorkingPeakSet::new(peaks),
            isotopic_model,
            scorer,
            fit_filter,
            scaling_method: TIDScalingMethod::default(),
            max_missed_peaks,
        }
    }
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak>,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > IsotopicPatternFitter<C> for DeconvoluterType<C, I, S, F>
{
    fn fit_theoretical_isotopic_pattern_with_params(&mut self, peak: PeakKey, charge: i32, params: IsotopicPatternParams) -> IsotopicFit {
        let mz = self.peaks.get(&peak).mz();
        let mut tid = self
            .isotopic_model
            .isotopic_cluster(mz, charge, params.charge_carrier, params.truncate_after, params.ignore_below);
        let (keys, missed_peaks) = self.peaks.match_theoretical(&tid, Tolerance::PPM(10.0));
        let exp = self.peaks.collect_for(&keys);
        self.scaling_method.scale(&exp, &mut tid);
        let score = self.scorer.score(&exp, &tid);
        IsotopicFit::new(keys, peak, tid, charge, score, missed_peaks as u16)
    }

    fn has_peak(&mut self, mz: f64, error_tolerance: Tolerance) -> PeakKey {
        let (peak, _missed) = self.peaks.has_peak(mz, error_tolerance);
        peak
    }

    fn between(&mut self, m1: f64, m2: f64) -> Range<usize> {
        self.peaks.between(m1, m2)
    }

    fn get_peak(&self, key: PeakKey) -> &C {
        self.peaks.get(&key)
    }

    fn create_key(&mut self, mz: f64) -> PeakKey {
        let i = self.peaks.placeholders.create(mz);
        PeakKey::Placeholder(i)
    }

    fn peak_count(&self) -> usize {
        self.peaks.len()
    }

    fn subtract_theoretical_intensity(&mut self, fit: &IsotopicFit) {
        self.peaks.subtract_theoretical_intensity(fit)
    }
}

pub type AveragineDeconvoluter<'lifespan> = DeconvoluterType<
    CentroidPeak,
    CachingIsotopicModel<'lifespan>,
    MSDeconvScorer,
    MaximizingFitFilter,
>;

#[derive(Debug)]
pub struct GraphDeconvoluterType<
    C: CentroidLike + Clone + From<CentroidPeak>,
    I: IsotopicPatternGenerator,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
    > {
    pub inner: DeconvoluterType<C, I, S, F>,
    pub peak_graph: PeakDependenceGraph
}

impl<C: CentroidLike + Clone + From<CentroidPeak>, I: IsotopicPatternGenerator, S: IsotopicPatternScorer, F: IsotopicFitFilter> GraphDeconvoluterType<C, I, S, F> {
    fn solve_subgraph_top(&mut self, cluster: DependenceCluster, fits: Vec<(FitRef, IsotopicFit)>) {
        if let Some(best_fit_key) = cluster.best_fit() {
            let (_, _fit) = fits.iter().find(|(k, _)| {
                *k == *best_fit_key
            }).unwrap_or_else(|| {
                panic!("Failed to locate a solution {:?}", best_fit_key);
            });
            todo!("Create peak")
        }
    }
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak>,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > IsotopicPatternFitter<C> for GraphDeconvoluterType<C, I, S, F> {

    fn fit_theoretical_isotopic_pattern_with_params(&mut self, peak: PeakKey, charge: i32, params: IsotopicPatternParams) -> IsotopicFit {
        self.inner.fit_theoretical_isotopic_pattern_with_params(peak, charge, params)
    }

    fn has_peak(&mut self, mz: f64, error_tolerance: Tolerance) -> PeakKey {
        self.inner.has_peak(mz, error_tolerance)
    }

    fn between(&mut self, m1: f64, m2: f64) -> Range<usize> {
        self.inner.between(m1, m2)
    }

    fn get_peak(&self, key: PeakKey) -> &C {
        self.inner.get_peak(key)
    }

    fn create_key(&mut self, mz: f64) -> PeakKey {
        self.inner.create_key(mz)
    }

    fn peak_count(&self) -> usize {
        self.inner.peak_count()
    }

    fn subtract_theoretical_intensity(&mut self, fit: &IsotopicFit) {
        self.inner.subtract_theoretical_intensity(fit)
    }
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak>,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > RelativePeakSearch<C> for GraphDeconvoluterType<C, I, S, F> {}

impl<
        C: CentroidLike + Clone + From<CentroidPeak>,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > ExhaustivePeakSearch<C> for GraphDeconvoluterType<C, I, S, F> {}

impl<
        C: CentroidLike + Clone + From<CentroidPeak>,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > GraphDeconvolution<C> for GraphDeconvoluterType<C, I, S, F> {

    fn add_fit_dependence(&mut self, fit: IsotopicFit) {
        if fit.experimental.is_empty() {
            return
        }
        let start = self.get_peak(*fit.experimental.first().unwrap()).mz();
        let end = self.get_peak(*fit.experimental.last().unwrap()).mz();
        self.peak_graph.add_fit(fit, start, end)
    }

    fn select_best_disjoint_subgraphs(&mut self) {
        self.peak_graph.find_non_overlapping_intervals();

        let solutions = self.peak_graph.solutions(SubgraphSolverMethod::Greedy);
        let _acc: Vec<_> = solutions.into_iter().map(|(cluster, fits)| {
            self.solve_subgraph_top(cluster, fits)
        }).collect();
        todo!()
    }
}



pub type GraphAveragineDeconvoluter<'lifespan> = GraphDeconvoluterType<
    CentroidPeak,
    CachingIsotopicModel<'lifespan>,
    MSDeconvScorer,
    MaximizingFitFilter,
>;




#[cfg(test)]
mod test {
    use std::fs;
    use std::io;

    use flate2::bufread::GzDecoder;

    use mzdata::MzMLReader;
    use mzpeaks::prelude::*;

    use crate::isotopic_model::IsotopicModels;
    use crate::scorer::MSDeconvScorer;

    use super::*;

    #[test]
    fn test_mut() {
        let peaks = vec![
            CentroidPeak::new(300.0, 150.0, 0),
            CentroidPeak::new(301.007, 5.0, 1),
        ];
        let peaks = MZPeakSetType::new(peaks);
        let mut task = AveragineDeconvoluter::new(
            peaks,
            IsotopicModels::Peptide.into(),
            MSDeconvScorer::default(),
            MaximizingFitFilter::default(),
            1,
        );
        let p = PeakKey::Matched(0);
        let fit1 = task.fit_theoretical_isotopic_pattern(p, 1);
        let fit2 = task.fit_theoretical_isotopic_pattern(p, 2);
        assert!(fit1.score > fit2.score);
    }

    #[test]
    fn test_fit_all() {
        let peaks = vec![
            CentroidPeak::new(300.0 - 1.007, 3.0, 0),
            CentroidPeak::new(300.0, 150.0, 2),
            CentroidPeak::new(301.007, 5.0, 3),
        ];
        let peaks = MZPeakSetType::new(peaks);
        let mut task = AveragineDeconvoluter::new(
            peaks,
            IsotopicModels::Peptide.into(),
            MSDeconvScorer::default(),
            MaximizingFitFilter::default(),
            1,
        );
        let solution_space =
            task.find_all_peak_charge_pairs(300.0, Tolerance::PPM(10.0), (1, 8), 1, 1, true);
        assert_eq!(solution_space.len(), 8);
        let n_matched = solution_space
            .iter()
            .map(|(k, _)| k.is_matched() as i32)
            .sum::<i32>();
        assert_eq!(n_matched, 7);
        let n_placeholders = solution_space
            .iter()
            .map(|(k, _)| k.is_placeholder() as i32)
            .sum::<i32>();
        assert_eq!(n_placeholders, 1);
    }

    #[test]
    fn test_file() -> io::Result<()> {
        let decoder = GzDecoder::new(io::BufReader::new(fs::File::open(
            "./tests/data/20150710_3um_AGP_001_29_30.mzML.gz",
        )?));
        let mut reader = MzMLReader::new(decoder);
        let scan = reader.next().unwrap();
        let centroided = scan.into_centroid().unwrap();

        let mut deconvoluter = AveragineDeconvoluter::new(
            centroided.peaks,
            IsotopicModels::Glycopeptide.into(),
            MSDeconvScorer::default(),
            MaximizingFitFilter::new(10.0),
            3,
        );

        let fits = deconvoluter.step_deconvolve(
            Tolerance::PPM(10.0),
            (1, 8),
            1,
            1,
            IsotopicPatternParams::default()
        );

        let best_fit = fits.iter().max().unwrap();
        assert_eq!(best_fit.charge, 4);
        assert_eq!(best_fit.missed_peaks, 0);
        assert!((best_fit.score - 2897.064989589997).abs() < 1e-3);
        assert_eq!(fits.len(), 3686);
        Ok(())
    }
}
