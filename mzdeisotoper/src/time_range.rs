use std::{error::Error, str::FromStr, fmt::Display, num::ParseFloatError, ops::Range};

use mzdeisotope::interval::Span1D;


#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeRange {
    pub start: f64,
    pub end: f64,
}

impl TimeRange {
    pub fn new(start: f64, end: f64) -> Self { Self { start, end } }
}

impl Span1D for TimeRange {
    type DimType = f64;

    fn start(&self) -> Self::DimType {
        self.start
    }

    fn end(&self) -> Self::DimType {
        self.end
    }
}

impl Default for TimeRange {
    fn default() -> Self {
        Self {
            start: 0.0,
            end: f64::INFINITY,
        }
    }
}

#[derive(Debug)]
pub enum TimeRangeParseError {
    MalformedStart(ParseFloatError),
    MalformedEnd(ParseFloatError),
}

impl Display for TimeRangeParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeRangeParseError::MalformedStart(e) => {
                write!(f, "Failed to parse time range start {e}")
            }
            TimeRangeParseError::MalformedEnd(e) => {
                write!(f, "Failed to parse time range end {e}")
            }
        }
    }
}

impl Error for TimeRangeParseError {}

impl FromStr for TimeRange {
    type Err = TimeRangeParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut tokens = if s.contains(' ') {
            s.split(' ')
        } else if s.contains(':') {
            s.split(':')
        } else if s.contains('-') {
            s.split('-')
        } else {
            s.split(' ')
        };
        let start_s = tokens.next().unwrap();
        let start_t = if start_s == "" {
            0.0
        } else {
            match start_s.parse() {
                Ok(val) => val,
                Err(e) => return Err(TimeRangeParseError::MalformedStart(e)),
            }
        };
        let end_s = tokens.next().unwrap();
        let end_t = if end_s == "" {
            f64::INFINITY
        } else {
            match end_s.parse() {
                Ok(val) => val,
                Err(e) => return Err(TimeRangeParseError::MalformedEnd(e)),
            }
        };
        Ok(TimeRange {
            start: start_t,
            end: end_t,
        })
    }
}

impl From<Range<f64>> for TimeRange {
    fn from(value: Range<f64>) -> Self {
        Self::new(value.start, value.end)
    }
}

impl From<(f64, f64)> for TimeRange {
    fn from(value: (f64, f64)) -> Self {
        Self::new(value.0, value.1)
    }
}