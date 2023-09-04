mod util;
use util::merge;
use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::{process::Command, path::PathBuf}; // Run programs
use std::path::Path;
use test_case::test_case;

#[test_case("derive", "1")]
#[test_case("extreme_long_code", "8820")]
#[test_case("extreme_long_code_reset", "500")]
#[test_case("extreme_long_name", "0")]
#[test_case("extreme_many_procs", "0")]
#[test_case("matrix_array", "1")]
#[test_case("matrix_ccnot", "1")]
#[test_case("matrix_complex", "0")]
#[test_case("matrix_decimal", "0")]
#[test_case("measure_twice", "0")]
#[test_case("teleport_diff_line", "0")]
#[test_case("teleport_one_line", "0")]
fn tests_fixed_output(name: &str, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("tests").join("input").join(name.to_string() + ".isq");
    fixed_output(path, res)
}

#[test_case("arith/mul", "6")]
#[test_case("bell", "0")]
#[test_case("bernstein_oracle_func", &merge(&["1", "1", "0"]))]
#[test_case("bv", "55")]
#[test_case("deutsch-jozsa", "0")]
#[test_case("ipe", "867893")]
#[test_case("oracle/bernstein", &merge(&["1", "1", "0"]))]
#[test_case("preserve_gphase", &merge(&["1", "0"]))]
#[test_case("teleport", "1")]
fn examples_fixed_output(name: &str, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("..").join("examples").join(name.to_string() + ".isq");
    fixed_output(path, res)
}

#[test_case("use_imported_proc", "1")]
#[test_case("use_imported_var", "1")]
#[test_case("shadow_imported_var", "2")]
#[test_case("qualified_name", "1")]
fn import_test(name: &str, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("..").join("frontend").join("test").join("input").join(name.to_string() + ".isq");
    fixed_output(path, res)
}

fn fixed_output(path: PathBuf, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg("--debug").arg(path.to_str().to_owned().unwrap());
    cmd.assert().success().stderr(predicate::str::contains(res));
    Ok(())
}

#[test_case("same_qubit", "is used twice")]
fn runtime_test(name: &str, syndrome: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file_name = "runtime_".to_string() + name + ".isq";
    let path = Path::new("tests").join("input").join(file_name);
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg("--debug").arg(path.to_str().to_owned().unwrap());
    cmd.assert().failure().stdout(predicate::str::contains(syndrome));
    Ok(())
}

#[test_case("grover")]
#[test_case("qubit")]
#[test_case("random")]
#[test_case("repeat_until_success")]
#[test_case("rfs")]
#[test_case("simon")]
#[test_case("tomography")]
fn expect_no_error(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("..").join("examples").join(name.to_string() + ".isq");
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg(path.to_str().to_owned().unwrap());
    cmd.assert().success().stderr(predicate::str::is_empty());
    Ok(())
}
