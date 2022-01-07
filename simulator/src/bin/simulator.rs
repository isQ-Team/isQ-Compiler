use isq_simulator::{
    devices::{checked::CheckedDevice, naive::NaiveSimulator},
    facades::qir::context::{get_current_context, make_context_current, QIRContext},
};
use libloading::*;

use clap::Parser;

extern crate env_logger;

#[macro_use]
extern crate std;
use std::io::Write;
use std::{cell::RefCell, ffi::OsString, rc::Rc};
extern crate isq_simulator;

#[derive(Parser, Debug)]
#[clap(about, version, author)]
struct SimulatorArgs {
    #[clap(index = 1, parse(from_os_str))]
    qir_shared_library: OsString,
    #[clap(short, long, default_value = "isq_simulator_entry")]
    entrypoint: String,
}

type SimulatorEntry = extern "C" fn() -> ();
fn main() -> std::io::Result<()> {
    /*env_logger::Builder::new()
    .format(|buf, record| {
        writeln!(
            buf,
            "{}:{} {} [{}] - {}",
            record.file().unwrap_or("unknown"),
            record.line().unwrap_or(0),
            chrono::Local::now().format("%Y-%m-%dT%H:%M:%S"),
            record.level(),
            record.args()
        )
    }).parse_default_env().init();*/
    env_logger::init();

    let args: SimulatorArgs = SimulatorArgs::parse();
    // initialize context.
    let device = CheckedDevice::new(NaiveSimulator::new());
    let context = QIRContext::new(
        Box::new(device),
        Box::new(|s| {
            println!("{}", s);
        }),
    );
    make_context_current(Rc::new(RefCell::new(context)));
    let library = unsafe { Library::new(args.qir_shared_library) }.unwrap();
    unsafe {
        let proc = library
            .get::<SimulatorEntry>(args.entrypoint.as_bytes())
            .unwrap();
        (proc)();
    }

    let ctx_ = get_current_context();
    let ctx = ctx_.borrow_mut();
    let res = ctx.get_classical_resource_manager();
    res.leak_check();
    return Ok(());
}
