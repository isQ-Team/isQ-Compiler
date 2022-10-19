mod util;
use util::LINE_ENDING;
use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::{process::Command}; // Run programs
use std::path::Path;
use test_case::test_case;

#[test_case("init_equal", &("4".to_string()+LINE_ENDING+"5"+LINE_ENDING+"6"))]
#[test_case("init_long", &("4".to_string()+LINE_ENDING+"5"))]
#[test_case("init_short", &("4".to_string()+LINE_ENDING+"5"))]
#[test_case("init_zero", &("4".to_string()+LINE_ENDING+"5"+LINE_ENDING+"6"))]
#[test_case("length", &("5".to_string()+LINE_ENDING+"5"))]
#[test_case("measure", &("4".to_string()+LINE_ENDING+"5"))]
fn test_array(name: &str, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let source_file = name.to_string() + ".isq";
    let folder = Path::new("tests").join("input").join("array");
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg("--debug").arg(folder.join(source_file).to_str().to_owned().unwrap());
    cmd.assert().success().stderr(predicate::str::contains(res));
    Ok(())
}