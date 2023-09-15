mod util;
use util::{merge, LINE_ENDING};
use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::process::Command; // Run programs
use std::path::Path;
use test_case::test_case;

#[test_case("add_sub_equal", &merge(&["7", "4", "4", "2"]))]
#[test_case("classic_add_sub", &merge(&["7", "-1"]))]
#[test_case("classic_and_or", &merge(&["0", "2"]))]
#[test_case("classic_bitwise_logic", &merge(&["8", "13", "5"]))]
#[test_case("classic_bool_to_int", "3")]
#[test_case("classic_comment", "10")]
#[test_case("classic_double", &("4".to_string()+LINE_ENDING+"6.283185"))]
#[test_case("classic_empty_statement", "0")]
#[test_case("classic_equal", &merge(&["0", "1"]))]
#[test_case("classic_local", &merge(&["789", "123", "456"]))]
#[test_case("classic_mod_pow", &merge(&["1", "-1", "49"]))]
#[test_case("classic_mul_div", &merge(&["32", "2"]))]
#[test_case("classic_nested_comment", "114514")]
#[test_case("classic_nested_region", "2")]
#[test_case("classic_recursion", "3628800")]
#[test_case("classic_shift", &merge(&["12", "2"]))]
#[test_case("classic_unitary", &merge(&["2", "-114514", "1"]))]
#[test_case("compare", &merge(&["1", "0", "1", "1"]))]
#[test_case("double_compare", &merge(&["1", "0", "1", "1"]))]
#[test_case("double_to_int", &merge(&["1", "-1", "1"]))]
#[test_case("int_to_bool", &merge(&["1", "0", "3", "2"]))]
#[test_case("fun_arg", &merge(&["1.1", "2"]))]
#[test_case("global", &merge(&["7.3", "4", "8"]))]
#[test_case("pow", &("8".to_string()+LINE_ENDING+"5.289"))]
#[test_case("switch", &merge(&["5"]))]
fn test_classic(name: &str, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let source_file = name.to_string() + ".isq";
    let folder = Path::new("tests").join("input").join("classic");
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg("--debug").arg(folder.join(source_file).to_str().to_owned().unwrap());
    cmd.assert().success().stderr(predicate::str::contains(res));
    Ok(())
}