use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::{process::Command}; // Run programs
use std::path::Path;
use test_case::test_case;

#[test_case("array")]
#[test_case("range_init_high")]
#[test_case("range_init_high_step")]
#[test_case("two_array")]
#[test_case("two_range")]
#[test_case("two_range_unequal")]
#[test_case("two_range_no_hi")]
fn test_bundle(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let source_file = name.to_string() + ".isq";
    let folder = Path::new("tests").join("input").join("qcis");
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("compile").arg("--target").arg("qcis").arg(folder.join(source_file).to_str().to_owned().unwrap());
    cmd.assert().success().stderr(predicate::str::is_empty());
    Command::new("diff")
            .arg(folder.join(name.to_string() + ".qcis").to_str().to_owned().unwrap())
            .arg(folder.join(name.to_string() + "_golden.qcis").to_str().to_owned().unwrap())
            .assert().success().stdout(predicate::str::is_empty());
    Ok(())
}