use std::{process::{Command, Stdio}, io::Write, ffi::OsStr};

pub fn exec_command_with_decorator<S: AsRef<OsStr>, F: FnOnce(&mut Command)->()>(root: &str, cmd: &str, args: &[S], sin: &[u8], child_decorator: F)->std::io::Result<Vec<u8>>{
    let path = match root{
        "" => String::from(cmd),
        s => format!("{}/bin/{}", s, cmd)
    };
    let mut child = Command::new(path);
    child.args(args)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::inherit());
    child_decorator(&mut child);
    let mut child = child.spawn()?;
    let mut stdin = child.stdin.take().expect("Failed to open stdin");
    let sin = sin.to_owned();
    std::thread::spawn(move || {
        stdin.write_all(&sin).expect("Failed to write to stdin");
    });
    let output = child.wait_with_output().expect("Failed to read stdout");
    Ok(output.stdout)
}
pub fn exec_command<S: AsRef<OsStr>>(root: &str, cmd: &str, args: &[S], sin: &[u8])->std::io::Result<Vec<u8>>{
    exec_command_with_decorator(root, cmd, args, sin,|_x|{})
}
pub fn exec_command_text<S: AsRef<OsStr>>(root: &str, cmd: &str, args: &[S], sin: &str)->std::io::Result<String>{
    let output = exec_command(root, cmd, args, sin.as_bytes())?;
    Ok(String::from_utf8_lossy(&output).into())
}

pub fn raw_exec_command<S: AsRef<OsStr>>(root: &str, cmd: &str, args: &[S])->std::io::Result<()>{
    let path = match root{
        "" => String::from(cmd),
        s => format!("{}/bin/{}", s, cmd)
    };
    let mut child = Command::new(path)
    .args(args)
    .stdin(Stdio::inherit())
    .stdout(Stdio::inherit())
    .stderr(Stdio::inherit())
    .spawn()?;
    let exit = child.wait()?;
    std::process::exit(exit.code().unwrap_or(1));
}