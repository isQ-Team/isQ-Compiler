use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::process::Command; // Run programs

#[test]
fn unrecognized_command() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("unknown");
    cmd.assert().failure().stderr(predicate::str::contains("which wasn't expected"));
    Ok(())
}

#[test]
fn help_command() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("-h");
    cmd.assert().success().stdout(predicate::str::contains("Print help information"));
    Ok(())
}

<<<<<<< HEAD
=======

/*
>>>>>>> merge
#[test]
fn nonexist_file() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("isqc")?;
    cmd.arg("compile").arg("non_exist.isq");
    cmd.assert().failure().stderr(predicate::str::contains("No such file or directory"));
    Ok(())
}
<<<<<<< HEAD
=======
*/
>>>>>>> merge
