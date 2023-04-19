use alloc::{string::String, vec::Vec, borrow::ToOwned};
use itertools::Itertools;
use serde::{Serialize, Deserialize};

use crate::qdevice::QDevice;

//pub const QCIS_ROUTE_BIN_PATH : Option<&'static str> = option_env!("ISQ_ROOT");

pub const QCIS_ROUTE_BIN_PATH : Option<&'static str> = option_env!("QCIS_ROUTE_BIN_PATH");
#[inline]
pub fn qcis_route_bin_path()->&'static str{
    QCIS_ROUTE_BIN_PATH.unwrap()
}

#[derive(Serialize, Deserialize)]
struct QCISImport{
    ty: String,
    q: Vec<usize>
}

#[derive(Serialize, Deserialize)]
struct QCISConfig{
    qbit_num: usize,
    topo: Vec<(usize, usize)>,
    init_map: String,
    qcis: Option<String>
}

fn generate_qcis_program_isqir(config: &QCISConfig, program: &[QCISImport])->String{
    let mut lines = Vec::new();
    let predefined_gates = [
        ("X","__quantum__qis__x__body", 1),
        ("Y","__quantum__qis__y__body", 1),
        ("Z","__quantum__qis__z__body", 1),
        ("H","__quantum__qis__h__body", 1),
        ("S","__quantum__qis__s__body", 1),
        ("T","__quantum__qis__t__body", 1),
        ("Sinv","__quantum__qis__s__adj", 1),
        ("Tinv","__quantum__qis__t__adj", 1),
        ("CZ","__quantum__qis__cz", 2),
        ("X2M","__quantum__qis__x2m", 1),
        ("X2P","__quantum__qis__x2p", 1),
        ("Y2M","__quantum__qis__y2m", 1),
        ("Y2P","__quantum__qis__y2p", 1),
        ("SWAP","__quantum__qis__swap", 2),
    ];
    for gate in predefined_gates.iter(){
        lines.push(format!("isq.defgate @qcis.{} {{definition = [#isq.gatedef<type = \"qir\", value = @{}>]}} : !isq.gate<{}>", gate.0, gate.1, gate.2));
        lines.push(format!("func.func private @{}({})", gate.1, (0..gate.2).into_iter().map(|_| "!isq.qir.qubit").join(", ")));
    }
    let nq = config.qbit_num;
    lines.push("isq.declare_qop @__isq__builtin__measure : [1] () -> i1".to_owned());
    lines.push("func.func @__isq__entry() {".to_owned());
    lines.push(format!("    %Q = memref.alloc() : memref<{}x!isq.qstate>", nq));
    let mut ssa = 1;
    for inst in program.iter(){
        let mut operands = Vec::new();
        let use_gate = ssa;
        ssa +=1;
        for q in inst.q.iter(){
            operands.push((q-1, ssa, ssa+1));
            ssa+=2;
        }
        lines.push(format!("    // {} {}", inst.ty, operands.iter().map(|x| format!("Q{}", x.0)).join(" "),));
        for (q, ssa_in, ssa_out) in operands.iter().copied(){
            lines.push(format!("    %{} = affine.load %Q[{}] : memref<{}x!isq.qstate>", ssa_in, q, nq));
        }
        if inst.ty == "M"{
            let m_ret = ssa;
            ssa+=1;
            lines.push(format!("    {}, %{} = isq.call_qop @__isq__builtin__measure({}) : [{}]()->i1",
                operands.iter().map(|x| format!("%{}", x.2)).join(", "),
                m_ret,
                operands.iter().map(|x| format!("%{}", x.1)).join(", "),
                operands.len()
            ));
        }else{
            lines.push(format!("    %{} = isq.use @qcis.{} : !isq.gate<{}>", use_gate, inst.ty, operands.len()));
            lines.push(format!("    {} = isq.apply %{}({}) : !isq.gate<{}>",
                operands.iter().map(|x| format!("%{}", x.2)).join(", "),
                use_gate,
                operands.iter().map(|x| format!("%{}", x.1)).join(", "),
                operands.len()
            ));
        }
        
        for (q, ssa_in, ssa_out) in operands.iter().copied(){
            lines.push(format!("    affine.store %{}, %Q[{}] : memref<{}x!isq.qstate>", ssa_out, q, nq));
        }
    }
    lines.push("    return".to_owned());
    lines.push("}".to_owned());
    lines.join("\n")
}

pub fn run_qcis_route(code: String)->String{
    extern crate std;
    extern crate serde_json;
    use std::{process::{Command, Stdio}, io::Write};
    use std::io::Read;
    let qcis_configuration = std::env::var("QCIS_ROUTE_CONFIG");//.expect("qcis routing configuration not speficied.");
    if let Ok(s) = qcis_configuration {
        let mut config_file = std::fs::File::open(s).expect("qcis config open failed");
        let mut config = String::new();
        config_file.read_to_string(&mut config).expect("qcis read failed");
        drop(config_file);
        let mut config_json = serde_json::from_str::<QCISConfig>(&config).expect("qcis json parse failed of schema invalid");
        config_json.qcis = Some(code);
        let input = serde_json::to_string(&config_json).expect("internal qcis error");
        let qcis_root_bin = qcis_route_bin_path();
        //let qcis_root_bin = std::env::var("ISQ_ROOT").expect("QCIS_ROUTE_BIN_PATH not defined at compile time!");
        let mut child = Command::new( qcis_root_bin)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect(&format!("qcis router failed to start: {}", qcis_root_bin));
        let mut stdin = child.stdin.take().expect("failed to open qcis route stdin");
        std::thread::spawn(move || {
            stdin.write_all(input.as_bytes()).expect("failed to write to qcis router");
        });
        let output = child.wait_with_output().expect("Failed to read stdout");
        output.status.exit_ok().expect("qcis router failed to exit. Please check.");
        let output = String::from_utf8(output.stdout).expect("qcis router yielded invalid output");
        let qcis_program : Vec<QCISImport> = serde_json::from_str(&output).expect(&format!("qcis router yielded bad json output {}", &output));
        generate_qcis_program_isqir(&config_json, &qcis_program)
    }else{
        return code;
    }
    
}

pub struct QCISCodegen{
    qcis_qubit_counter: usize,
    finalized: bool,
    generated_code: Vec<String>
}

impl QCISCodegen{
    pub fn new()->Self{
        Self{qcis_qubit_counter: 0, finalized: false, generated_code: vec![]}
    }
    pub fn append_op(&mut self, op: &str, args: &[&usize]){
        let args_separated = args.iter().map(|x| format!("Q{}", **x+1)).join(" ");
        self.generated_code.push(format!("{} {}", op, args_separated));
    }
    pub fn finalize_route(&mut self){
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
            SInv,TInv,CZ,X2M,X2P,Y2M,Y2P, QcisFinalize
        ]
    }

    fn measure(&mut self, qubit: &Self::Qubit) -> bool {
        self.append_op("M", &[qubit]);
        return false;
    }

    fn qop(&mut self, op_type: crate::qdevice::QuantumOp, qubits: &[&Self::Qubit], parameters: &[f64]) {
        use crate::qdevice::QuantumOp::*;
        if let QcisFinalize = op_type{
            if !self.finalized{
                self.finalized=true;
                self.finalize_route();
            }
            
            return;
        }
        if let Rz = op_type{
            let op = format!("RZ");
            let args_separated = qubits.iter().map(|x| format!("Q{}", **x+1)).join(" ");
            self.generated_code.push(format!("{} {} {}", op, args_separated, parameters[0]));
            return;
        }
        let op_name = match op_type{
            X=>"X", Y=>"Y", Z=>"Z",
            H=>"H", S=>"S", T=>"T",
            SInv=>"SD", TInv=>"TD", CZ=>"CZ",
            X2P=>"X2P", X2M=>"X2M",
            Y2P=>"Y2P", Y2M=>"Y2M",
            _ => panic!("bad op type {:?}", op_type)
        };
        self.append_op(op_name, qubits);
    }
    
}
