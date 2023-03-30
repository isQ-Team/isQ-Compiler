mod util;
use util::LINE_ENDING;
use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::{process::Command}; // Run programs
use std::path::Path;
use test_case::test_case;

#[test_case("scf_block", "0")]
#[test_case("scf_break", "2")]
#[test_case("scf_break_for", "2")]
#[test_case("scf_continue", "4")]
#[test_case("scf_continue_for", "7")]
#[test_case("scf_else_if", "2")]
#[test_case("scf_empty_block", "0")]
#[test_case("scf_for", "10")]
#[test_case("scf_for_neg", "10")]
#[test_case("scf_for_array", &("2".to_string()+LINE_ENDING+"3"+LINE_ENDING+"4"))]
#[test_case("scf_for_raw_array", &("2".to_string()+LINE_ENDING+"3"+LINE_ENDING+"4"))]
#[test_case("scf_for_array_zero", &("2".to_string()+LINE_ENDING+"3"+LINE_ENDING+"4"))]
#[test_case("scf_for_if", "4")]
#[test_case("scf_for_step", "4")]
#[test_case("scf_if", "2")]
#[test_case("scf_if_break", "2")]
#[test_case("scf_if_break_block", "2")]
#[test_case("scf_if_break_for", "2")]
#[test_case("scf_if_no_else", "3")]
#[test_case("scf_while", "6")]
#[test_case("scf_while_no_brace", "16")]
fn test_scf(name: &str, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let source_file = name.to_string() + ".isq";
    let folder = Path::new("tests").join("input").join("scf");
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg("--debug").arg(folder.join(source_file).to_str().to_owned().unwrap());
    cmd.assert().success().stderr(predicate::str::contains(res));
    Ok(())
}