mod util;
use util::merge;
use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::{process::Command}; // Run programs
use std::path::Path;
use test_case::test_case;

#[test_case("example", &merge(&["7", "3", "2", "3"]))]
#[test_case("three_qubit_adder", &merge(&["7", "5", "12"]))]
#[test_case("two_qubit_adder", &merge(&["2", "1", "3"]))]
#[test_case("uninitialized_array", &merge(&["1", "2"]))]
fn test_oracle(name: &str, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let source_file = name.to_string() + ".isq";
    let folder = Path::new("tests").join("input").join("oracle");
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg("--debug").arg(folder.join(source_file).to_str().to_owned().unwrap());
    cmd.assert().success().stderr(predicate::str::contains(res));
    Ok(())
}
