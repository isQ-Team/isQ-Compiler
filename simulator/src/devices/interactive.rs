// Interactive simulator. Printing all operations, and reading measurement results from input stream.

use alloc::collections::BTreeSet;
use crate::qdevice::QDevice;



pub trait MeasurementResultReader{
    fn next_measurement_result(&mut self) -> bool;
}

pub struct AllZeroMeasurementResults;
impl MeasurementResultReader for AllZeroMeasurementResults{
    fn next_measurement_result(&mut self) -> bool {
        false
    }
}
pub struct AllOneMeasurementResults;
impl MeasurementResultReader for AllOneMeasurementResults{
    fn next_measurement_result(&mut self) -> bool {
        true
    }
}

pub struct InteractiveSimulator<T: MeasurementResultReader>{
    existing_qubits: BTreeSet<usize>,
    qubit_counter: usize,
    reader: T
}


impl<T: MeasurementResultReader> InteractiveSimulator<T>{
    pub fn new(reader: T)->Self{
        InteractiveSimulator{
            existing_qubits: BTreeSet::new(),
            qubit_counter: 0,
            reader
        }
    }
}

impl<T: MeasurementResultReader> QDevice for InteractiveSimulator<T>{
    type Qubit = usize;
    fn alloc_qubit(&mut self)->Self::Qubit {
        let q = self.qubit_counter;
        self.existing_qubits.insert(q);
        self.qubit_counter+=1;
        q
    }

    fn free_qubit(&mut self, qubit: Self::Qubit) {
        if self.existing_qubits.remove(&qubit) == false{
            panic!("Qubit {} is not allocated", qubit);
        }
    }

    fn supported_quantum_ops()->alloc::vec::Vec<crate::qdevice::QuantumOp> {
        use crate::qdevice::QuantumOp::*;
        vec![
            Reset, X, Y, Z, H, S, T,
            CNOT, CZ, Swap, CCNOT,
            Rx, Ry, Rz, U3
        ]
    }

    fn qop(&mut self, op_type: crate::qdevice::QuantumOp, qubits: &[&Self::Qubit], parameters: &[f64]) {
        info!("Performing {:?} on qubits {:?} with parameters {:?}", op_type, qubits, parameters);
    }

    fn measure(&mut self, qubit: &Self::Qubit)->bool {
        let r = self.reader.next_measurement_result();
        info!("Measurement on qubit {} = {}", qubit, r);
        r
    }
}