mod util;
use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::{process::Command}; // Run programs
use std::path::Path;
use test_case::test_case;

#[test_case("array_negative", "Error: index out of bound")]
#[test_case("array_out_of_bound", "Error: index out of bound")]
#[test_case("divide_0", "Error: divide 0")]
#[test_case("assert_false", "Error: failed assertion")]
fn test_assert_stdout(name: &str, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let source_file = name.to_string() + ".isq";
    let folder = Path::new("tests").join("input").join("assert");
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg("--debug").arg(folder.join(source_file).to_str().to_owned().unwrap());
    cmd.assert().failure().stdout(predicate::str::contains(res));
    Ok(())
}

#[test_case("length_mismatch", "are cast incompatible")]
fn test_assert_stderr(name: &str, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let source_file = name.to_string() + ".isq";
    let folder = Path::new("tests").join("input").join("assert");
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg("--debug").arg(folder.join(source_file).to_str().to_owned().unwrap());
    cmd.assert().failure().stderr(predicate::str::contains(res));
    Ok(())
}

#[test_case("projection_assert")]
#[test_case("shor")]
fn test_assert_no_error(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let source_file = name.to_string() + ".isq";
    let folder = Path::new("tests").join("input").join("assert");
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg(folder.join(source_file).to_str().to_owned().unwrap());
    cmd.assert().success().stderr(predicate::str::is_empty());
    Ok(())
}
