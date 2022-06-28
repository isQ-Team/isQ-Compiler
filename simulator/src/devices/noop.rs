use crate::qdevice::QDevice;

pub struct NoopDevice{
    qubit_counter: usize,
}

impl NoopDevice{
    pub fn new()->Self{
        Self{qubit_counter: 0}
    }
}

impl QDevice for NoopDevice{
    type Qubit = usize;

    fn alloc_qubit(&mut self) -> Self::Qubit {
        let qubit = self.qubit_counter;
        self.qubit_counter+=1;
        qubit
    }

    fn free_qubit(&mut self, qubit: Self::Qubit) {
        // no-op.
    }

    fn supported_quantum_ops(&self) -> alloc::vec::Vec<crate::qdevice::QuantumOp> {
        use crate::qdevice::QuantumOp::*;
        vec![
            X,Y,Z,H,S,T,
            SInv,TInv,CZ,X2M,X2P,Y2M,Y2P, QCIS_Finalize, U3, AnySQ, Reset
        ]
    }

    fn measure(&mut self, qubit: &Self::Qubit) -> bool {
        return false;
    }

    fn qop(&mut self, op_type: crate::qdevice::QuantumOp, qubits: &[&Self::Qubit], parameters: &[f64]) {
        return;
    }
    
}