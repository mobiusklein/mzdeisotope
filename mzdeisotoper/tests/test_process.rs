use std::{error::Error, process::Command};

use assert_cmd::prelude::*;
use predicates::prelude::*;

#[test]
fn test_file_missing() -> Result<(), Box<dyn Error>> {
    let mut cmd = Command::cargo_bin("mzdeisotoper")?;

    cmd.arg("not_real.mzML").arg("-o").arg("-");
    cmd.assert().failure().stderr(predicate::str::contains(
        "The system cannot find the file specified",
    ));
    Ok(())
}

#[test]
fn test_malformed_time_range() -> Result<(), Box<dyn Error>> {
    let mut cmd = Command::cargo_bin("mzdeisotoper")?;

    cmd.arg("not_real.mzML").arg("-o").arg("-").args(["-r a-z"]);
    cmd.assert().failure().stderr(predicate::str::contains(
        "Failed to parse time range end invalid float literal",
    ));

    let mut cmd = Command::cargo_bin("mzdeisotoper")?;

    cmd.arg("not_real.mzML").arg("-o").arg("-").args(["-r -a"]);
    cmd.assert().failure().stderr(predicate::str::contains(
        "Failed to parse time range end invalid float literal",
    ));

    Ok(())
}

#[test]
fn test_run_subset() -> Result<(), Box<dyn Error>> {
    let mut cmd = Command::cargo_bin("mzdeisotoper")?;
    cmd.env("RUST_LOG", "info");
    cmd.arg("./tests/data/batching_test.mzML")
        .args(["-o", "-", "-r", "120-120.1"]);
    let result = cmd.assert().success();
    result
        .stderr(predicate::str::contains("MS1 Spectra: 1"))
        .stderr(predicate::str::contains("MSn Spectra: 21"))
        .stderr(predicate::str::contains("| Scans=1 Time=120.004"));

    Ok(())
}
