mod util;
use util::LINE_ENDING;
use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::{process::Command}; // Run programs
use std::path::Path;
use test_case::test_case;

#[test_case("classic_add_sub", &("7".to_string()+LINE_ENDING+"-1"))]
#[test_case("classic_and_or", &("0".to_string()+LINE_ENDING+"2"))]
#[test_case("classic_bitwise_logic", &("8".to_string()+LINE_ENDING+"13"+LINE_ENDING+"5"))]
#[test_case("classic_bool_to_int", "3")]
#[test_case("classic_comment", "10")]
#[test_case("classic_double", &("4".to_string()+LINE_ENDING+"6.283185"))]
#[test_case("classic_empty_statement", "0")]
#[test_case("classic_equal", &("0".to_string()+LINE_ENDING+"1"))]
#[test_case("classic_local", &("789".to_string()+LINE_ENDING+"123"+LINE_ENDING+"456"))]
#[test_case("classic_mod_pow", &("1".to_string()+LINE_ENDING+"49"))]
#[test_case("classic_mul_div", &("32".to_string()+LINE_ENDING+"2"))]
#[test_case("classic_nested_comment", "114514")]
#[test_case("classic_nested_region", "2")]
#[test_case("classic_recursion", "3628800")]
#[test_case("classic_shift", &("12".to_string()+LINE_ENDING+"2"))]
#[test_case("classic_unitary", &("2".to_string()+LINE_ENDING+"-114514"+LINE_ENDING+"1"))]
fn test_assert(name: &str, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let source_file = name.to_string() + ".isq";
    let folder = Path::new("tests").join("input").join("classic");
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg("--debug").arg(folder.join(source_file).to_str().to_owned().unwrap());
    cmd.assert().success().stderr(predicate::str::contains(res));
    Ok(())
}