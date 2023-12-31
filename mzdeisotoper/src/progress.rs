use std::ops::{Add, AddAssign};


#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct ProgressRecord {
    pub ms1_peaks: usize,
    pub msn_peaks: usize,
    pub precursors_defaulted: usize,
    pub precursor_charge_state_mismatch: usize,
    pub ms1_spectra: usize,
    pub msn_spectra: usize,
}

impl Add for ProgressRecord {
    type Output = ProgressRecord;

    fn add(self, rhs: Self) -> Self::Output {
        let mut dup = self.clone();
        dup += rhs;
        dup
    }
}

impl AddAssign for ProgressRecord {
    fn add_assign(&mut self, rhs: Self) {
        self.ms1_peaks += rhs.ms1_peaks;
        self.msn_peaks += rhs.msn_peaks;
        self.precursors_defaulted += rhs.precursors_defaulted;
        self.precursor_charge_state_mismatch += rhs.precursor_charge_state_mismatch;
        self.ms1_spectra += rhs.ms1_spectra;
        self.msn_spectra += rhs.msn_spectra;
    }
}
