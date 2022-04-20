use alloc::{string::String, vec::Vec};
use itertools::Itertools;

use crate::qdevice::QDevice;

pub const QCIS_ROUTE_BIN_PATH : Option<&'static str> = option_env!("QCIS_ROUTE_BIN_PATH");

#[inline]
pub fn qcis_route_bin_path()->&'static str{
    QCIS_ROUTE_BIN_PATH.expect("QCIS_ROUTE_BIN_PATH not defined at compile time!")
}

pub fn run_qcis_route(code: String)->String{
    extern crate std;
    extern crate serde_json;
    use serde_json::Value;
    use std::{process::{Command, Stdio}, io::Write};
    use std::io::Read;
    let qcis_configuration = std::env::var("QCIS_ROUTE_CONFIG").expect("qcis routing configuration not speficied.");
    let mut config_file = std::fs::File::open(qcis_configuration).expect("qcis config open failed");
    let mut config = String::new();
    config_file.read_to_string(&mut config).expect("qcis read failed");
    drop(config_file);
    let mut config_json = serde_json::from_str::<Value>(&config).expect("qcis json failed");
    if let Value::Object(obj) = &mut config_json{
        obj.insert(String::from("qcis"), Value::String(code));
    }else{
        panic!("qcis config schema invalid");
    }
    let input = serde_json::to_string(&config_json).expect("internal qcis error");
    let mut child = Command::new(qcis_route_bin_path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("qcis router failed to start.");
    let mut stdin = child.stdin.take().expect("failed to open qcis route stdin");
    std::thread::spawn(move || {
        stdin.write_all(input.as_bytes()).expect("failed to write to qcis router");
    });
    let output = child.wait_with_output().expect("Failed to read stdout");
    output.status.exit_ok().expect("qcis router failed to exit. Please check.");
    String::from_utf8(output.stdout).expect("qcis router yielded invalid output")
}

pub struct QCISCodegen{
    qcis_qubit_counter: usize,
    generated_code: Vec<String>
}

impl QCISCodegen{
    pub fn new()->Self{
        Self{qcis_qubit_counter: 0, generated_code: vec![]}
    }
    pub fn append_op(&mut self, op: &str, args: &[&usize]){
        let args_separated = args.iter().map(|x| format!("Q{}", x)).join(" ");
        self.generated_code.push(format!("{} {}", op, args_separated));
    }
    pub fn finalize_route(self){
        extern crate std;
        let output = run_qcis_route(self.generated_code.join("\n"));
        std::println!("{}", output);
    }
}


impl QDevice for QCISCodegen{
    type Qubit = usize;

    fn alloc_qubit(&mut self) -> Self::Qubit {
        let qubit = self.qcis_qubit_counter;
        self.qcis_qubit_counter+=1;
        qubit
    }

    fn free_qubit(&mut self, qubit: Self::Qubit) {
        // no-op.
    }

    fn supported_quantum_ops(&self) -> alloc::vec::Vec<crate::qdevice::QuantumOp> {
        use crate::qdevice::QuantumOp::*;
        vec![
            X,Y,Z,H,S,T,
            SInv,TInv,CZ,X2M,X2P,Y2M,Y2P
        ]
    }

    fn measure(&mut self, qubit: &Self::Qubit) -> bool {
        self.append_op("M", &[qubit]);
        return false;
    }

    fn qop(&mut self, op_type: crate::qdevice::QuantumOp, qubits: &[&Self::Qubit], parameters: &[f64]) {
        use crate::qdevice::QuantumOp::*;
        let op_name = match op_type{
            X=>"X", Y=>"Y", Z=>"Z",
            H=>"H", S=>"S", T=>"T",
            SInv=>"SD", TInv=>"TD", CZ=>"CZ",
            X2P=>"X2P", X2M=>"X2M",
            Y2P=>"Y2P", Y2M=>"Y2M",
            _ => panic!("bad op type")
        };
        self.append_op(op_name, qubits);
    }
    
}