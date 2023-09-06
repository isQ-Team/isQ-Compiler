mod util;
use util::merge;
use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::process::Command; // Run programs
use std::path::Path;
use test_case::test_case;

#[test_case("bcastc", 3, &merge(&["1", "1", "1"]))]
#[test_case("qpe", 3, &merge(&["23"]))]
#[test_case("teleport", 2, &merge(&["0"]))]
fn test_distributed(name: &str, np: i32, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let source_file = name.to_string() + ".isq";
    let folder = Path::new("..").join("examples").join("dqc");
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg("--debug").arg("-n").arg(np.to_string()).arg(folder.join(source_file).to_str().to_owned().unwrap());
    cmd.assert().success().stderr(predicate::str::contains(res));
    Ok(())
}

#[test_case("approximate_or", 4)]
#[test_case("bcast", 3)]
#[test_case("poorcat", 3)]
fn distributed_no_error(name: &str, np: i32) -> Result<(), Box<dyn std::error::Error>> {
    let source_file = name.to_string() + ".isq";
    let folder = Path::new("..").join("examples").join("dqc");
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg("-n").arg(np.to_string()).arg(folder.join(source_file).to_str().to_owned().unwrap());
    cmd.assert().success().stderr(predicate::str::is_empty());
    Ok(())
}