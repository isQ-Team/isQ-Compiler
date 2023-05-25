mod util;
use util::merge;
use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::{process::Command}; // Run programs
use std::path::Path;
use test_case::test_case;

#[test_case("init", &merge(&["4", "5", "6"]))]
#[test_case("length", &merge(&["5", "5", "5", "5"]))]
#[test_case("measure", &merge(&["4", "5"]))]
#[test_case("slice", &merge(&["3", "0", "3", "6"]))]
fn test_array(name: &str, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let source_file = name.to_string() + ".isq";
    let folder = Path::new("tests").join("input").join("array");
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg("--debug").arg(folder.join(source_file).to_str().to_owned().unwrap());
    cmd.assert().success().stderr(predicate::str::contains(res));
    Ok(())
}

#[test_case("measure_statement")]
fn array_no_error(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let source_file = name.to_string() + ".isq";
    let folder = Path::new("tests").join("input").join("array");
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg(folder.join(source_file).to_str().to_owned().unwrap());
    cmd.assert().success().stderr(predicate::str::is_empty());
    Ok(())
}