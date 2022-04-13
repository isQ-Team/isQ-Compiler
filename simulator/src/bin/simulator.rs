use isq_simulator::{
    devices::{checked::CheckedDevice, naive::NaiveSimulator},
    facades::qir::context::{get_current_context, make_context_current, QIRContext}, qdevice::QDevice,
};
#[cfg(feature = "cuda")]
use isq_simulator::devices::cuda::QSimKernelSimulator;
use libloading::*;

use clap::{Parser, ArgEnum, PossibleValue};

extern crate env_logger;

#[macro_use]
extern crate std;
use std::io::Write;
use std::{cell::RefCell, ffi::OsString, rc::Rc};
extern crate isq_simulator;


use clap::ArgGroup;
#[derive(Parser, Debug)]
#[clap(about, version, author)]
#[clap(group(
    ArgGroup::new("simulator_type")
        .required(true)
        .args(&["naive", "cuda"]),
))]
struct SimulatorArgs {
    #[clap(index = 1, parse(from_os_str))]
    qir_shared_library: OsString,
    #[clap(short, long, default_value = "isq_simulator_entry")]
    entrypoint: String,
    #[clap(long)]
    naive: bool,
    #[clap(long)]
    cuda: Option<usize>
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
    let device: Box<dyn QDevice<Qubit=usize>> = {
        if args.naive{
            Box::new(CheckedDevice::new(NaiveSimulator::new()))
        }else if let Some(cap) = args.cuda{
            #[cfg(not(feature = "cuda"))]
            panic!("Simulator is built with `cuda` feature disabled.");
            #[cfg(feature = "cuda")]
            Box::new(CheckedDevice::new(QSimKernelSimulator::new(cap)))
        }else{
            unreachable!();
        }
    };
    // initialize context.
    let context = QIRContext::new(
        device,
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
