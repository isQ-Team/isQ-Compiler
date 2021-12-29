use libloading::*;

use clap::Parser;
#[macro_use]
extern crate std;
use std::ffi::OsString;
extern crate isq_simulator;

#[derive(Parser, Debug)]
#[clap(about, version, author)]
struct SimulatorArgs{
    #[clap(index = 1, parse(from_os_str))]
    qir_shared_library: OsString,
    #[clap(short, long, default_value = "isq_simulator_entry")]
    entrypoint: String,
}

type SimulatorEntry = extern "C" fn ()->();
fn main()->std::io::Result<()>{
    let args: SimulatorArgs = SimulatorArgs::parse();
    let library = unsafe {Library::new(args.qir_shared_library)}.unwrap();
    unsafe{
        let proc = library.get::<SimulatorEntry>(args.entrypoint.as_bytes()).unwrap();
        (proc)();
    }
    return Ok(());
}