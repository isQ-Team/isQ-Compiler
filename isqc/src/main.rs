
extern crate clap;

mod exec;
mod frontend;
mod error;
use std::{fs::File, io::{Read, Write}, path::{Path, PathBuf}, ffi::OsStr};

use clap::*;
use error::*;
use std::str::FromStr;
use tempfile::tempfile;
use crate::frontend::resolve_isqc1_output;

fn opt_level(s: &str) -> Result<(), String> {
    usize::from_str(s)
        .map(|o| o>=0 && o<=3)
        .map_err(|e| e.to_string())
        .and_then(|result| match result {
            true => Ok(()),
            false => Err(format!(
                "Optimization level should be within 0-3"
            )),
        })
}

#[derive(Parser)]
#[clap(author, version, about)]
pub struct Arguments {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
pub enum Commands{
    Compile{
        input: String,
        #[clap(long, short)]
        output: Option<String>,
        #[clap(long, short='O', validator = opt_level)]
        opt_level: Option<usize>,
    },
    Simulate{
        #[clap(required(true))]
        qir_object: String,
        #[clap(long)]
        cuda: Option<usize>
    },
    Exec{
        #[clap(multiple_occurrences(true), required(true))]
        exec_command: Vec<String>
    }
}

fn main()->miette::Result<()> {
    let cli = Arguments::parse();
    let root = std::env::var("ISQV2_ROOT").map_err(|_| NoISQv2RootError)?;
    
    match cli.command{
        Commands::Compile{input, output, opt_level}=>{
            let input_path = Path::new(&input);
            
            if input_path.extension()!=Some(OsStr::new("isq")){
                return Err(BadExtensionError)?;
            }
            let output = output.map_or_else(||{
                input_path.with_extension("so")
            }, |x| {PathBuf::from(x)});
            
            let mut f = File::open(input_path).map_err(IoError)?;
            let mut buf = String::new();
            f.read_to_string(&mut buf).unwrap();
            let mlir = exec::exec_command_text::<&str>(&root, "isqc1", &[], &buf).map_err(ioErrorWhen("Calling isqc1"))?;
            let resolved_mlir = resolve_isqc1_output(input_path.file_name().unwrap().to_str().unwrap(), &buf, &mlir)?;
            let llvm_mlir = exec::exec_command_text(&root, "isq-opt", &[
                "-pass-pipeline=canonicalize,cse,isq-fold-constant-decorated-gates,isq-decompose-known-gates-qsd,isq-expand-decomposition,isq-lower-to-qir-rep,cse,canonicalize,isq-lower-qir-rep-to-llvm,canonicalize,cse,symbol-dce,llvm-legalize-for-export",
                "--mlir-print-debuginfo"
            ], &resolved_mlir).map_err(ioErrorWhen("Calling isq-opt"))?;
            let llvm = exec::exec_command_text(&root, "mlir-translate", &["--mlir-to-llvmir"], &llvm_mlir).map_err(ioErrorWhen("Calling mlir-translate"))?;
            // linking with stub. This step we use byte output.
            let linked_llvm = exec::exec_command(&root, "llvm-link", &[
                format!("-"),
                format!("{}/share/isq-simulator/isq-simulator.bc", &root)
            ], llvm.as_bytes()).map_err(ioErrorWhen("Calling llvm-link"))?;
            let mut opt_args: Vec<String> = Vec::new();
            if let Some(o) = opt_level{
                opt_args.push(format!("-O{}", o));
            }
            let optimized_llvm = exec::exec_command(&root, "opt", &opt_args, &linked_llvm).map_err(ioErrorWhen("Calling opt"))?;
            let compiled_obj = exec::exec_command(&root, "llc", &["-filetype=obj"], &optimized_llvm).map_err(ioErrorWhen("Calling llc"))?;
            // create obj file.
            let mut tmpfile = tempfile::NamedTempFile::new().map_err(ioErrorWhen("Creating tempfile"))?;
            tmpfile.write_all(&compiled_obj).map_err(IoError)?;
            tmpfile.flush().map_err(IoError)?;
            // link obj file.
            let linked_obj = exec::exec_command(&root, "lld", &["-flavor", "gnu", "-shared", tmpfile.path().as_os_str().to_str().unwrap(), "-o", "-"], &[]).map_err(ioErrorWhen("Calling ld.lld"))?;
            drop(tmpfile);
            // Finally, write obj into output.
            let mut fout = File::create(output).map_err(IoError)?;
            fout.write_all(&linked_obj).map_err(IoError)?;
        }
        Commands::Simulate{qir_object, cuda}=>{
            let qir_object = if qir_object.starts_with("/"){
                qir_object
            }else{
                format!("./{}", qir_object)
            };
            let mut v = vec!["-e".into(), "__isq__entry".into()];
            if let Some(x)=cuda{
                v.push("--cuda".into());
                v.push(format!("{}", x));
            }else{
                v.push("--naive".into());
            }
            v.push(qir_object);
            exec::raw_exec_command(&root, "simulator", &v).map_err(IoError)?;
        }
        Commands::Exec{exec_command}=>{
            exec::raw_exec_command(&root, &exec_command[0], &exec_command[1..], ).map_err(IoError)?;
        }
    }
    return Ok(())
}
