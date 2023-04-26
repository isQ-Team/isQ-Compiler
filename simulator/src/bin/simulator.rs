use isq_simulator::{
    devices::{checked::CheckedDevice, naive::NaiveSimulator, sq2u3::SQ2U3Device, noop::NoopDevice},
    facades::qir::{context::{get_current_context, make_context_current, QIRContext, RANK_REF}}, qdevice::QDevice,
};

#[cfg(feature = "qcis")]
use isq_simulator::sim::qcis;

use libloading::*;

use clap::Parser;

extern crate env_logger;
use std::{env::{set_var, self}, sync::{mpsc, Mutex, Arc}};
use log::debug;

#[macro_use]
extern crate std;
use std::{io::{Read, Write}, path::Path, fs::File, collections::HashMap};
use std::{ffi::OsString, thread};
extern crate isq_simulator;

use clap::ArgGroup;
#[derive(Parser, Debug)]
#[clap(about, version, author)]
#[clap(group(
    ArgGroup::new("simulator_type")
        .required(true)
        .args(&["naive", "cuda", "qcisgen", "noop", "qcis"]),
))]
struct SimulatorArgs {
    #[clap(index = 1, parse(from_os_str))]
    qir_shared_library: OsString,
    #[clap(short, long, default_value = "__isq_simulator_entry")]
    entrypoint: String,
    #[clap(long)]
    naive: bool,
    #[clap(long)]
    cuda: Option<usize>,
    #[clap(long)]
    qcisgen: bool,
    #[clap(long)]
    noop: bool,
    #[clap(long)]
    qcis: bool,
    #[clap(long)]
    shots: Option<i64>,
    #[clap(long)]
    debug: bool,
    #[clap(long, short)]
    int_par: Option<Vec<i64>>,
    #[clap(long, short)]
    double_par: Option<Vec<f64>>,
    #[clap(long, short, default_value = "1")]
    np: i64
}


type SimulatorEntry = extern "C" fn(x_alloc_ptr: *const i64, x_align_ptr: *const i64, x_offset: i64, x_size: i64, x_strides: i64,
                                    y_alloc_ptr: *const f64, y_align_ptr: *const f64, y_offset: i64, y_size: i64, y_strides: i64,
                                    rank: i64) -> ();
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
    

    let args: SimulatorArgs = SimulatorArgs::parse();

    if args.debug{
        if env::var("RUST_LOG").is_err(){
            set_var("RUST_LOG", "simulator=debug, isq_simulator::facades::qir::shim=debug");
        }
        
    }

    env_logger::builder().format(|buf, record| {
        writeln!(buf, "{}", record.args())
    }).init();

    let shots = match args.shots {
        Some(v) => v,
        _ => 1
    };

    if args.qcis{
        #[cfg(feature = "qcis")]
        {
            let input_path = Path::new(&args.qir_shared_library);
            let mut f = File::open(input_path)?;
            let mut buf = String::new();
            f.read_to_string(&mut buf).unwrap();
            qcis::sim(buf, shots);
            return Ok(());
        }
        #[cfg(not(feature = "qcis"))]
        {
            panic!("QCIS plugin not compiled!")
        }

    }

    let par_int = match args.int_par {
        Some(x) => x,
        None => vec![]
    };
    let par_double = match args.double_par{
        Some(x) => x,
        None => vec![]
    };

    let mut res_map: HashMap<String, i32> = HashMap::new();
    for i in 0..shots{
        debug!("{}th simulation print:", i);
        let device: Box<dyn QDevice<Qubit=usize>> = {
            if args.naive{
                Box::new(CheckedDevice::new(SQ2U3Device::new(NaiveSimulator::new())))
            }else if let Some(cap) = args.cuda{
                #[cfg(not(feature = "cuda"))]
                panic!("Simulator is built with `cuda` feature disabled. {}", cap);
                #[cfg(feature = "cuda")]
                {
                    use isq_simulator::devices::cuda::QSimKernelSimulator;
                    Box::new(CheckedDevice::new(SQ2U3Device::new(QSimKernelSimulator::new(cap))))
                }
                
            }else if args.qcisgen{
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
            args.np,
        );
        
        make_context_current(Arc::new(Mutex::new(context)));

        unsafe {
            let mut handles = vec![];
            for rank in 0..args.np {
                let (tx_int, rx_int) = mpsc::channel();
                let (tx_double, rx_double) = mpsc::channel();
                let (tx_osstring, rx_osstring) = mpsc::channel::<OsString>();
                let (tx_string, rx_string) = mpsc::channel::<String>();
                let handle = thread::spawn(move || {
                    RANK_REF.with(|r| *r.borrow_mut() = rank);
                    let mut par_int = vec![];
                    for received in rx_int {
                        par_int.push(received);
                    }
                    let par_int_ptr = par_int.as_ptr();

                    let mut par_double = vec![];
                    for received in rx_double {
                        par_double.push(received);
                    }
                    let par_double_ptr = par_double.as_ptr();

                    let osstring = rx_osstring.recv().unwrap();
                    let library = Library::new(osstring).unwrap();

                    let entrypoint = rx_string.recv().unwrap();
                    let proc = library.get::<SimulatorEntry>(entrypoint.as_bytes()).unwrap();
                    (proc)(par_int_ptr, par_int_ptr, 0, 0, 0, par_double_ptr, par_double_ptr, 0, 0, 0, rank);
                });
                for v in &par_int {
                    tx_int.send(*v).unwrap();
                }
                for v in &par_double {
                    tx_double.send(*v).unwrap();
                }
                tx_osstring.send(args.qir_shared_library.clone()).unwrap();
                tx_string.send(args.entrypoint.clone()).unwrap();
                handles.push(handle);
            }
            for handle in handles {
                handle.join().unwrap();
            }
        }
        
        let ctx_ = get_current_context();
        let mut ctx = ctx_.lock().unwrap();

        let r = ctx.get_device_mut().get_measure_res();
        let count = res_map.entry(r).or_insert(0);
        *count += 1;
        
        let res = ctx.get_classical_resource_manager();
        res.leak_check();
    }
    if !args.qcisgen{
        println!("{:?}", res_map);
    }
    return Ok(());
}
