use std::{fs, io};

use mzdata;
use mzdata::prelude::*;
use mzdeisotope::DeconvolvedSolutionPeak;
use mzpeaks::{CentroidPeak, MassPeakSetType};

use mzsvg::v2::ColorCycle;
use mzsvg::{self, CentroidSeries, Group, PlotSeries, SeriesDescription};

#[derive(Debug, Clone)]
struct IsotopicEnvelopSeries {
    peaks: MassPeakSetType<DeconvolvedSolutionPeak>,
    colors: ColorCycle,
    description: SeriesDescription,
}

impl PlotSeries<f64, f32> for IsotopicEnvelopSeries {
    fn description(&self) -> &SeriesDescription {
        &self.description
    }

    fn description_mut(&mut self) -> &mut SeriesDescription {
        &mut self.description
    }

    fn to_svg(&self, canvas: &mzsvg::Canvas<f64, f32>) -> Group {
        let mz_peaks = self
            .peaks
            .iter()
            .map(|p| CentroidPeak::new(p.mz(), p.intensity(), 0))
            .collect();

        let mut root = Group::new();
        root = root.add(CentroidSeries::new(mz_peaks, self.description.clone()).to_svg(canvas));

        let mut cycle = self.colors.clone();
        for p in self.peaks.iter() {
            let env = p.envelope.iter().map(|i| i.as_centroid()).collect();
            let color = cycle.next().unwrap();
            root = root.add(
                CentroidSeries::new(
                    env,
                    SeriesDescription::new(format!("{:0.2}, {}", p.neutral_mass, p.charge), color),
                )
                .to_svg(canvas),
            );
        }
        root
    }

    fn slice_x(&mut self, start: f64, end: f64) {
        self.peaks = self
            .peaks
            .iter()
            .filter(|p| start <= p.mz() && p.mz() <= end)
            .cloned()
            .collect();
    }

    fn slice_y(&mut self, start: f32, end: f32) {
        let points = self
            .peaks
            .iter()
            .filter(|p| (p.intensity() >= start) && (p.intensity() <= end))
            .cloned()
            .collect();
        self.peaks = points;
    }
}

fn main() -> io::Result<()> {
    println!("Hello, world!");

    Ok(())
}
