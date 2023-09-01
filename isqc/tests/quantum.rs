mod util;
use util::merge;
use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::process::Command; // Run programs
use std::path::Path;
use test_case::test_case;

#[test_case("add", &merge(&["3", "6", "1"]))]
#[test_case("call_array", &merge(&["1", "1"]))]
#[test_case("cnot", &merge(&["0", "3"]))]
#[test_case("init", &merge(&["2", "1", "5"]))]
#[test_case("init_array", &merge(&["1", "1"]))]
#[test_case("init_default", &merge(&["0"]))]
#[test_case("init_ket", &merge(&["2"]))]
#[test_case("sub", &merge(&["5", "2", "7"]))]
#[test_case("switch", &merge(&["0"]))]
#[test_case("x", &merge(&["3", "7"]))]
fn test_quantum(name: &str, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let source_file = name.to_string() + ".isq";
    let folder = Path::new("tests").join("input").join("quantum");
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg("--debug").arg(folder.join(source_file).to_str().to_owned().unwrap());
    cmd.assert().success().stderr(predicate::str::contains(res));
    Ok(())
}
