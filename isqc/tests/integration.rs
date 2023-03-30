mod util;
use util::LINE_ENDING;
use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::{process::Command, path::PathBuf}; // Run programs
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
#[test_case("reset_twice", "0")]
#[test_case("teleport_diff_line", "0")]
#[test_case("teleport_one_line", "0")]
fn tests_fixed_output(name: &str, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("tests").join("input").join(name.to_string() + ".isq");
    fixed_output(path, res)
}

#[test_case("bell", "0")]
#[test_case("bernstein", &("1".to_string()+LINE_ENDING+"1"+LINE_ENDING+"0"))]
#[test_case("bernstein_oracle_func", &("1".to_string()+LINE_ENDING+"1"+LINE_ENDING+"0"))]
#[test_case("ipe", "867893")]
#[test_case("mul", "6")]
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

#[test_case("empty_gate", "Syntax Error: unexpected token")]
#[test_case("keyword_as_identifier", "Syntax Error: unexpected token")]
#[test_case("mat_not_square", "bad matrix shape")]
#[test_case("mat_not_2_pow", "Syntax Error: unexpected token")]
#[test_case("not_utf8", "invalid byte sequence")]
#[test_case("other_lang", "Syntax Error: tokenizing failed")]
#[test_case("repeated_names", "isqv2::frontend::redefined_symbol")]
#[test_case("type_mismatch", "Type mismatch")]
#[test_case("undefined_symbol", "Undefined symbol")]
#[test_case("unknown_token", "Syntax Error: tokenizing failed")]
#[test_case("wrong_brackets", "Syntax Error: unexpected token")]
#[test_case("wrong_break", "Unexpected statement outside a loop")]
#[test_case("wrong_continue", "Unexpected statement outside a loop")]
#[test_case("wrong_for", "Syntax Error: unexpected token")]
#[test_case("wrong_ctrl_size", "Argument number mismatch")]
#[test_case("wrong_inv", "Syntax Error: unexpected token")]
#[test_case("wrong_size", "Argument number mismatch")]
fn syntax_test(name: &str, syndrome: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file_name = "syntax_".to_string() + name + ".isq";
    let path = Path::new("tests").join("input").join(file_name);
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("compile").arg(path.to_str().to_owned().unwrap());
    cmd.assert().failure().stderr(predicate::str::contains(syndrome));
    Ok(())
}

//#[test_case("out_of_range", "does not exist")]
#[test_case("same_qubit", "is used twice")]
fn runtime_test(name: &str, syndrome: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file_name = "runtime_".to_string() + name + ".isq";
    let path = Path::new("tests").join("input").join(file_name);
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg("--debug").arg(path.to_str().to_owned().unwrap());
    cmd.assert().success().stderr(predicate::str::contains(syndrome));
    Ok(())
}

#[test_case("grover")]
#[test_case("qubit")]
#[test_case("random")]
#[test_case("repeat_until_success")]
#[test_case("rfs")]
fn expect_no_error(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("..").join("examples").join(name.to_string() + ".isq");
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("run").arg(path.to_str().to_owned().unwrap());
    cmd.assert().success().stderr(predicate::str::is_empty());
    Ok(())
}
