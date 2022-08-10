use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::process::Command; // Run programs

#[test]
fn use_imported_proc() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("../../../bin/isqc")?;
    cmd.arg("run").arg("../frontend/test/input/use_imported_proc.isq");
    cmd.assert().success().stdout(predicate::str::contains("1"));
    Ok(())
}

#[test]
fn use_imported_var() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("../../../bin/isqc")?;
    cmd.arg("run").arg("../frontend/test/input/use_imported_var.isq");
    cmd.assert().success().stdout(predicate::str::contains("1"));
    Ok(())
}

#[test]
fn shadow_imported_var() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("../../../bin/isqc")?;
    cmd.arg("run").arg("../frontend/test/input/shadow_imported_var.isq");
    cmd.assert().success().stdout(predicate::str::contains("2"));
    Ok(())
}

#[test]
fn qualified_name() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("../../../bin/isqc")?;
    cmd.arg("run").arg("../frontend/test/input/qualified_name.isq");
    cmd.assert().success().stdout(predicate::str::contains("1"));
    Ok(())
}

