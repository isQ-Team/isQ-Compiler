use assert_cmd::prelude::*; // Add methods on commands
use assert_cmd::assert::Assert;
use miette::Diagnostic;
use predicates::prelude::*; // Used for writing assertions
use std::{process::Command, path::PathBuf}; // Run programs
use std::path::Path;
use test_case::test_case;
use thiserror::Error;

#[derive(Error, Debug, Diagnostic)]
#[error("ISQ_ROOT undefined.")]
#[diagnostic(
    code(isqv2::no_isqv2_root),
    help("This means something is wrong if you are calling from isqc entry.")
)]
pub struct NoISQv2RootError;

#[cfg(windows)]
pub const LINE_ENDING: &'static str = "\r\n";
#[cfg(not(windows))]
pub const LINE_ENDING: &'static str = "\n";


#[test_case("classic_add", "7")]
#[test_case("classic_comment", "10")]
#[test_case("classic_divide0", "inf")]
#[test_case("classic_double", &("4".to_string()+LINE_ENDING+"6.283185"))]
#[test_case("classic_if", "2")]
#[test_case("classic_local", &("789".to_string()+LINE_ENDING+"123"+LINE_ENDING+"456"))]
#[test_case("classic_neg", "-114514")]
#[test_case("classic_nested_comment", "114514")]
#[test_case("classic_recursion", "3628800")]
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
    let path = Path::new("isqc").join("tests").join("input").join(name.to_string() + ".isq");
    fixed_output(path, res)
}

#[test_case("bell", "0")]
#[test_case("bernstein", &("1".to_string()+LINE_ENDING+"1"+LINE_ENDING+"0"))]
#[test_case("ipe", "867893")]
#[test_case("preserve_gphase", &("1".to_string()+LINE_ENDING+"0"))]
#[test_case("teleport", "1")]
fn examples_fixed_output(name: &str, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("examples").join(name.to_string() + ".isq");
    fixed_output(path, res)
}

#[test_case("use_imported_proc", "1")]
#[test_case("use_imported_var", "1")]
#[test_case("shadow_imported_var", "2")]
#[test_case("qualified_name", "1")]
fn import_test(name: &str, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("frontend").join("test").join("input").join(name.to_string() + ".isq");
    fixed_output(path, res)
}

fn fixed_output(path: PathBuf, res: &str) -> Result<(), Box<dyn std::error::Error>> {
    let cmd = integration_test_template(path, "run").unwrap();
    cmd.success().stdout(predicate::str::contains(res));
    Ok(())
}

#[test_case("unknown_token", "Syntax Error: tokenizing failed")]
#[test_case("keyword_as_identifier", "Syntax Error: unexpected token")]
#[test_case("repeated_names", "isqv2::frontend::redefined_symbol")]
#[test_case("wrong_for", "Syntax Error: unexpected token")]
#[test_case("wrong_ctrl_size", "Argument number mismatch")]
#[test_case("wrong_inv", "Syntax Error: unexpected token")]
#[test_case("wrong_size", "Argument number mismatch")]
#[test_case("other_lang", "Syntax Error: tokenizing failed")]
#[test_case("not_utf8", "invalid byte sequence")]
#[test_case("empty_gate", "Syntax Error: unexpected token")]
#[test_case("wrong_brackets", "Syntax Error: unexpected token")]
#[test_case("type_mismatch", "Type mismatch")]
#[test_case("undefined_symbol", "Undefined symbol")]
#[test_case("mat_not_square", "bad matrix shape")]
#[test_case("mat_not_2_pow", "Syntax Error: unexpected token")]
fn syntax_test(name: &str, syndrome: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file_name = "syntax_".to_string() + name + ".isq";
    let path = Path::new("isqc").join("tests").join("input").join(file_name);
    let cmd = integration_test_template(path, "compile").unwrap();
    cmd.failure().stderr(predicate::str::contains(syndrome));
    Ok(())
}

#[test_case("out_of_range", "does not exist")]
#[test_case("same_qubit", "is used twice")]
fn runtime_test(name: &str, syndrome: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file_name = "runtime_".to_string() + name + ".isq";
    let path = Path::new("isqc").join("tests").join("input").join(file_name);
    let cmd = integration_test_template(path, "run").unwrap();
    cmd.success().stderr(predicate::str::contains(syndrome));
    Ok(())
}

#[test_case("grover")]
#[test_case("qubit")]
#[test_case("random")]
#[test_case("repeat_until_success")]
#[test_case("rfs")]
fn expect_no_error(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("examples").join(name.to_string() + ".isq");
    let cmd = integration_test_template(path, "run").unwrap();
    cmd.success().stderr(predicate::str::is_empty());
    Ok(())
}

fn integration_test_template(file_path: PathBuf, option: &str) -> Result<Assert, Box<dyn std::error::Error>> {
    let root = std::env::var("ISQ_ROOT").map_err(|_| NoISQv2RootError)?;
    let root_path = Path::new(&root);
    let isqc = root_path.join("bin").join("isqc");
    let mut cmd = Command::cargo_bin(isqc.to_str().to_owned().unwrap())?;
    cmd.arg(option).arg(root_path.join(file_path).to_str().to_owned().unwrap());
    Ok(cmd.assert())
}
