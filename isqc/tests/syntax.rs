use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::process::Command; // Run programs
use std::path::Path;
use test_case::test_case;

#[test_case("array_initialization", "Syntax Error: unexpected token")]
#[test_case("array_no_initialization", "Syntax Error: unexpected token")]
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
    let file_name = name.to_string() + ".isq";
    let path = Path::new("tests").join("input").join("syntax").join(file_name);
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("compile").arg(path.to_str().to_owned().unwrap());
    cmd.assert().failure().stderr(predicate::str::contains(syndrome));
    Ok(())
}