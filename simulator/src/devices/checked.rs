// Device wrapper to check sanity of the operation before calling the actual device.
// For example, no cloning theorem.
// Also the wrapper guarantees QDevice::Qubit is usize growing continuously.

use crate::qdevice::*;
use alloc::{vec::Vec, collections::BTreeMap};
pub struct CheckedDevice<Q: Clone, T: QDevice<Qubit = Q>> {
    pub device: T,
    pub qubit_counter: usize,
    pub qubit_mapping: BTreeMap<usize, Q>,
}

impl <Q: Clone, T: QDevice<Qubit = Q>> CheckedDevice<Q, T> {
    pub fn new(device: T) -> Self {
        Self {
            device,
            qubit_counter: 0,
            qubit_mapping: BTreeMap::new(),
        }
    }
}
impl <Q: Clone, T: QDevice<Qubit = Q>> QDevice for CheckedDevice<Q, T>{
    type Qubit = usize;
    fn alloc_qubit(&mut self) -> Self::Qubit {
        let qubit_id = self.qubit_counter;
        self.qubit_counter+=1;
        let real_qubit = self.device.alloc_qubit();
        self.qubit_mapping.insert(qubit_id, real_qubit);
        qubit_id
    }
    fn free_qubit(&mut self, qubit: Self::Qubit) {
        if let Some(real_qubit) = self.qubit_mapping.remove(&qubit) {
            self.device.free_qubit(real_qubit);
        } else {
            panic!("QDevice sanity check failed: qubit {} is not allocated", qubit);
        }
    }
    fn supported_quantum_ops() -> Vec<QuantumOp> {
        T::supported_quantum_ops()
    }
    fn qop(&mut self, op_type: QuantumOp, qubits: &[&Self::Qubit], parameters: &[f64]) {
        if qubits.len() != op_type.get_qubit_count() {
            panic!("Qubit count mismatch");
        }
        if parameters.len() != op_type.get_parameter_count() {
            panic!("Parameter count mismatch");
        }
        for i in 0..qubits.len(){
            for j in i+1..qubits.len(){
                if qubits[i]==qubits[j]{
                    panic!("Qubit argument #{} is used twice (next use: qubit argument #{})", i, j);
                }
            }
        }
        let m = &mut self.qubit_mapping;
        let real_qubits = qubits.iter().map(|x| m.get(&x).expect(&format!("Qubit #{} does not exist", x))).collect::<Vec<_>>();
        self.device.qop(op_type, &real_qubits, parameters)
    }
    fn measure(&mut self, x: &Self::Qubit) -> bool {
        let real_qubit = self.qubit_mapping.get(&x).expect(&format!("Qubit #{} does not exist", x));
        self.device.measure(&real_qubit)
    }
}