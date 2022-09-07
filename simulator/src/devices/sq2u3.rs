use alloc::collections::BTreeSet;
use num_complex::Complex64;

use crate::qdevice::{QDevice, QuantumOp};

pub struct SQ2U3Device<Q, T: QDevice<Qubit = Q>> (T, BTreeSet<QuantumOp>);



pub fn translate_sq_gate_to_matrix(op_type: QuantumOp, parameters: &[f64])->[[Complex64; 2]; 2]{
    use crate::qdevice::QuantumOp::*;
    let invsqrt2 = (0.5f64).sqrt();
    match op_type{
        U3=>{
            let theta = parameters[0];
            let phi = parameters[1];
            let lambda = parameters[2];
            let i = Complex64::i();
            let mat = [
                [
                    (-i*(phi+lambda)/2.0).exp() * (theta / 2.0).cos(),
                    -((i * (lambda-phi)/2.0).exp()) * (theta / 2.0).sin(),
                ],
                [
                    (i * (phi-lambda)/2.0).exp() * (theta / 2.0).sin(),
                    (i * (phi + lambda)/2.0).exp() * (theta / 2.0).cos(),
                ],
            ];
            mat
        }
        X=>[[0.0.into(), 1.0.into()], [1.0.into(), 0.0.into()]],
        Y=>[
            [0.0.into(), Complex64::new(0.0, 1.0)],
            [Complex64::new(0.0, -1.0), 0.0.into()],
        ],
        Z=>[[1.0.into(), 0.0.into()], [0.0.into(), (-1.0).into()]],
        H=>[
            [invsqrt2.into(), invsqrt2.into()],
            [invsqrt2.into(), (-invsqrt2).into()],
        ],
        S=>[
            [1.0.into(), 0.0.into()],
            [0.0.into(), Complex64::new(0.0, 1.0)],
        ],
        SInv=>[
            [1.0.into(), 0.0.into()],
            [0.0.into(), Complex64::new(0.0, -1.0)],
        ],
        T=>[
            [1.0.into(), 0.0.into()],
            [0.0.into(), Complex64::new(invsqrt2, invsqrt2)],
        ],
        TInv=>[
            [1.0.into(), 0.0.into()],
            [0.0.into(), Complex64::new(invsqrt2, -invsqrt2)],
        ],
        Rx => {
            let theta = parameters[0];
            let mat = [
                [
                    (theta / 2.0).cos().into(),
                    (-Complex64::i() * ((theta / 2.0).sin())).into(),
                ],
                [
                    (-Complex64::i() * (theta / 2.0).sin()).into(),
                    (theta / 2.0).cos().into(),
                ],
            ];
            mat
        },
        Ry => {
            let theta = parameters[0];
            let mat = [
                [(theta / 2.0).cos().into(), (-((theta / 2.0).sin())).into()],
                [(theta / 2.0).sin().into(), (theta / 2.0).cos().into()],
            ];
            mat
        }
        Rz => {
            let theta = parameters[0];
            let mat = [
                [(-Complex64::i() * theta / 2.0).exp(), 0.0.into()],
                [0.0.into(), (Complex64::i() * theta / 2.0).exp()],
            ];
            mat
        }
        // These are not really used.
        X2P => [
            [Complex64::new(invsqrt2, 0.0), Complex64::new(0.0, -invsqrt2)],
            [Complex64::new(0.0, -invsqrt2), Complex64::new(invsqrt2, 0.0)],
        ],
        X2M => [
            [Complex64::new(invsqrt2, 0.0), Complex64::new(0.0, invsqrt2)],
            [Complex64::new(0.0, invsqrt2), Complex64::new(invsqrt2, 0.0)],
        ],
        Y2P => [
            [Complex64::new(invsqrt2, 0.0), Complex64::new(-invsqrt2, 0.0)],
            [Complex64::new( invsqrt2, 0.0), Complex64::new(invsqrt2, 0.0)],
        ],
        Y2M => [
            [Complex64::new(invsqrt2, 0.0), Complex64::new(invsqrt2, 0.0)],
            [Complex64::new( -invsqrt2, 0.0), Complex64::new(invsqrt2, 0.0)],
        ],
        _=>panic!("bad sq {:?}", op_type)
    }
}
impl<Q: Eq, T: QDevice<Qubit = Q>> SQ2U3Device<Q, T>{
    pub fn new(dev: T)->Self{
        let supported = dev.supported_quantum_ops().iter().copied().collect();
        Self(dev, supported)
    }
}

impl<Q: Eq, T: QDevice<Qubit = Q>> QDevice for SQ2U3Device<Q, T>{

    type Qubit = Q;
    fn controlled_qop(
        &mut self,
        op_type: crate::qdevice::QuantumOp,
        controllers: &[&Self::Qubit],
        qubits: &[&Self::Qubit],
        parameters: &[f64],
    ) {
        if self.1.contains(&op_type){
            self.0.controlled_qop(op_type, controllers, qubits, parameters)
        }else if op_type.get_qubit_count()==1{
            let matrix = translate_sq_gate_to_matrix(op_type, parameters);
            // size assertion
            let params = unsafe {core::mem::transmute::<_, [f64; 8]>(matrix)};
            self.0.controlled_qop(QuantumOp::AnySQ, controllers, qubits, &params)
        }
    }
    

    fn alloc_qubit(&mut self) -> Self::Qubit {
        self.0.alloc_qubit()
    }

    fn free_qubit(&mut self, qubit: Self::Qubit) {
        self.0.free_qubit(qubit)
    }

    fn supported_quantum_ops(&self) -> alloc::vec::Vec<crate::qdevice::QuantumOp> {
        let mut ops = self.0.supported_quantum_ops();
        if ops.contains(&QuantumOp::AnySQ){
            use QuantumOp::*;
            let mut sq_ops = vec![H, S, T, X, Y, Z, U3, SInv, TInv, Rx, Ry, Rz, X2M, X2P, Y2M, Y2P];
            ops.append(&mut sq_ops);
        }
        ops.sort();
        ops.dedup();
        return ops
    }

    fn measure(&mut self, qubit: &Self::Qubit) -> bool {
        self.0.measure(qubit)
    }

    fn get_measure_res(&mut self) -> alloc::string::String {
        self.0.get_measure_res()
    }
    fn print_state(&self) { self.0.print_state(); }
}
