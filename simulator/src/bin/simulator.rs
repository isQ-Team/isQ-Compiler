use isq_simulator::{
    devices::{checked::CheckedDevice, naive::NaiveSimulator, sq2u3::SQ2U3Device, noop::NoopDevice},
    facades::qir::context::{get_current_context, make_context_current, QIRContext}, qdevice::QDevice,
};

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
        .args(&["naive", "cuda", "qcis", "noop"]),
))]
struct SimulatorArgs {
    #[clap(index = 1, parse(from_os_str))]
    qir_shared_library: OsString,
    #[clap(short, long, default_value = "isq_simulator_entry")]
    entrypoint: String,
    #[clap(long)]
    naive: bool,
    #[clap(long)]
    cuda: Option<usize>,
    #[clap(long)]
    qcis: bool,
    #[clap(long)]
    noop: bool,
    #[clap(long, short)]
    int_par: Option<Vec<i64>>,
    #[clap(long, short)]
    double_par: Option<Vec<f64>> 
}

type SimulatorEntry = extern "C" fn(x_alloc_ptr: *const i64, x_align_ptr: *const i64, x_offset: i64, x_size: i64, x_strides: i64,
                                    y_alloc_ptr: *const f64, y_align_ptr: *const f64, y_offset: i64, y_size: i64, y_strides: i64) -> ();
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
            Box::new(CheckedDevice::new(SQ2U3Device::new(NaiveSimulator::new())))
        }else if let Some(cap) = args.cuda{
            #[cfg(not(feature = "cuda"))]
            panic!("Simulator is built with `cuda` feature disabled.");
            #[cfg(feature = "cuda")]
            {
                use isq_simulator::devices::cuda::QSimKernelSimulator;
                Box::new(CheckedDevice::new(SQ2U3Device::new(QSimKernelSimulator::new(cap))))
            }
            
        }else if args.qcis{
            #[cfg(not(feature = "qcis"))]
            panic!("Simulator is built with `qcis` feature disabled.");
            #[cfg(feature = "qcis")]
            {
                use isq_simulator::devices::qcisgen::QCISCodegen;
                Box::new(CheckedDevice::new(QCISCodegen::new()))
            }
            
        }else if args.noop{
            Box::new(CheckedDevice::new(SQ2U3Device::new(NoopDevice::new())))
        }else {
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
    
    let par_int = match args.int_par {
        Some(x) => x,
        None => vec![]
    };
    let par_int_ptr = par_int.as_ptr();
    let par_double = match args.double_par{
        Some(x) => x,
        None => vec![]
    };
    let par_double_ptr = par_double.as_ptr();

    let library = unsafe { Library::new(args.qir_shared_library) }.unwrap();
    unsafe {
        let proc = library
            .get::<SimulatorEntry>(args.entrypoint.as_bytes())
            .unwrap();
        (proc)(par_int_ptr, par_int_ptr, 0, 0, 0, par_double_ptr, par_double_ptr, 0, 0, 0);
    }

    let ctx_ = get_current_context();
    let ctx = ctx_.borrow_mut();
    let res = ctx.get_classical_resource_manager();
    res.leak_check();
    return Ok(());
}
