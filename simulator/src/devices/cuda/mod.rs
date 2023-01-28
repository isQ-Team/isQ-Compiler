mod qsim_kernel;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use num_complex::Complex32;
use qsim_kernel::qstate;
use alloc::string::String;
use crate::{devices::cuda::qsim_kernel::qstate_cnot, qdevice::QDevice};

use self::qsim_kernel::{qstate_struct_size, qint_t, qstate_init, qstate_deinit, qstate_align_size, qamph_t, qstate_u3, qstate_measure, qstate_swap_to_msb_and_free, qstate_alloc, qstate_debug_amps};
pub struct QSimKernel{
    state: *mut qstate,
    capacity: usize,
    // Logical to physical
    qubit_map_l2p: BTreeMap<usize, usize>,
    // Physical to logical
    qubit_map_p2l: Vec<usize>,
    // id alloc
    next_id: usize,
    // measure res
    measure_res: String,
}

impl QSimKernel{
    pub fn new(capacity: usize)->Self{
        unsafe{
            let mem = alloc::alloc::alloc(alloc::alloc::Layout::from_size_align(qstate_struct_size() as usize, qstate_align_size() as usize).unwrap()) as *mut qstate;
            qstate_init(mem, capacity as qint_t);
            let qubit_map = BTreeMap::new();
            let mut qubit_map_inv = Vec::new();
            for _ in 0..capacity{
                qubit_map_inv.push(0);
            }
            QSimKernel { state: mem , capacity, qubit_map_l2p: qubit_map, qubit_map_p2l: qubit_map_inv, next_id: 0, measure_res: String::new()}
        }
    }
    pub fn size(&self)->usize{
        self.qubit_map_l2p.len()
    }
    pub fn capacity(&self)->usize{
        self.capacity
    }
    fn logical_to_physical(&self, q: usize)->usize{
        *self.qubit_map_l2p.get(&q).expect("Logical qubit released")
    }

    pub fn u3(&mut self, q: usize, mat: &[Complex32; 4]){
        let p = self.logical_to_physical(q);
        let mut mat_amps: [qamph_t; 8] = [0.0; 8];
        for (i, c) in mat.iter().enumerate(){
            mat_amps[2*i]=c.re;
            mat_amps[2*i+1]=c.im;
        }
        unsafe{
            qstate_u3(self.state, p as qint_t, mat_amps.as_mut_ptr());
        }
    }
    pub fn cnot(&mut self, q1: usize, q2: usize){
        assert_ne!(q1, q2);
        let p1 = self.logical_to_physical(q1);
        let p2 = self.logical_to_physical(q2);
        unsafe{
            qstate_cnot(self.state, p1 as _, p2 as _ );
        }
    }
    pub fn measure_reset(&mut self, q: usize, reset: bool, r: f32)->(bool, f32){
        let p = self.logical_to_physical(q);
        let mut out_prob: f32 = 0.0;
        let out = unsafe{
            qstate_measure(self.state, p as _, if reset {1} else {0}, r, &mut out_prob as *mut _)
        }!=0;
        (out, out_prob)
    }
    pub fn traceout(&mut self, q: usize, r: f32)->f32{
        let p = self.logical_to_physical(q);
        let mut out_prob: f32 = 0.0;
        let replaced_physical_qubit = unsafe{
            qstate_swap_to_msb_and_free(self.state, p as _, r, &mut out_prob)
        } as usize;
        let swapped_logical = self.qubit_map_p2l[replaced_physical_qubit];
        self.qubit_map_l2p.insert(swapped_logical, p);
        self.qubit_map_p2l[p]=swapped_logical;
        self.qubit_map_l2p.remove(&q);
        out_prob
    }
    pub fn alloc(&mut self)->usize{
        let id = self.next_id;
        self.next_id+=1;
        if self.size()>=self.capacity(){
            panic!("The number of qubits required exceeds capacity. Try a larger capacity.");
        }
        let new_physical = unsafe{
            qstate_alloc(self.state)
        } as usize;
        self.qubit_map_l2p.insert(id,new_physical);
        self.qubit_map_p2l[new_physical]=id;
        id

    }
    pub fn dump(&mut self){
        let mut v: Vec<Complex32> = Vec::new();
        let s = (1usize)<<(self.size());
        v.reserve(s);
        unsafe{
            qstate_debug_amps(self.state, v.as_mut_ptr() as *mut _, s as _);
            v.set_len(s);
        }
        let mut total = 0f32;
        for i in v.iter(){
            total+=i.norm_sqr();
        }
        trace!("Dump: {:?}", v);
        if total<1.0-(1e-3){
            panic!("bad dump");
        }
        if total>1.0+(1e-3){
            panic!("bad dump either");
        }
        
    }
}
impl Drop for QSimKernel{
    fn drop(&mut self){
        unsafe{
            qstate_deinit(self.state);
        }
    }
}


pub struct QSimKernelSimulator(QSimKernel);

impl QSimKernelSimulator{
    pub fn new(capacity: usize)->Self{
        trace!("Cuda simulator created with capacity {}", capacity);
        Self(QSimKernel::new(capacity))
    }
}

impl QDevice for QSimKernelSimulator{
    type Qubit = usize;
    fn alloc_qubit(&mut self)->Self::Qubit{
        let new_qubit = self.0.alloc();
        trace!("Allocating qubit {}", new_qubit);
        new_qubit
    }
    fn free_qubit(&mut self, qubit: Self::Qubit){
        let prob_zero = self.0.traceout(qubit, rand::random::<f32>());
        let prob_one = 0.0_f32.max(1.0-prob_zero);
        if prob_zero >= 1e-4 && prob_one >= 1e-4 {
            warn!(
                "Qubit {} is not in computational state! prob = ({}, {})",
                qubit, prob_zero, prob_one
            );
            warn!("This may be treated as an error in Q# environment. However we perform trace-out here.");
        }
        trace!("Freeing qubit {}", qubit);
    }
    fn supported_quantum_ops(&self) -> Vec<crate::qdevice::QuantumOp> {
        use crate::qdevice::QuantumOp::*;
        vec![Reset, CNOT, CZ, U3]
    }
    fn qop(
        &mut self,
        op_type: crate::qdevice::QuantumOp,
        qubits: &[&Self::Qubit],
        parameters: &[f64],
    ) {
        //self.0.dump();
        trace!(
            "Perform {:?}{:?} on {:?}",
            op_type,
            parameters,
            qubits
        );
        use crate::qdevice::QuantumOp::*;
        match op_type {
            Reset => {
                let (result, prob_zero) = self.0.measure_reset(*qubits[0], true, rand::random::<f32>());
                trace!(
                    "Measuring(reset!) qubit {} (prob = ({}, {})), yielding {}",
                    *qubits[0],
                    prob_zero,
                    1.0-prob_zero,
                    result
                );
            }
            CNOT => {
                self.0.cnot(*qubits[0], *qubits[1]);
            }
            AnySQ => {
                let p = parameters;
                let mat: [Complex32; 4] = [
                    Complex32::new(p[0] as f32, p[1] as f32), Complex32::new(p[2] as f32, p[3] as f32),
                    Complex32::new(p[4] as f32, p[5] as f32), Complex32::new(p[6] as f32, p[7] as f32)
                ];
                self.0.u3(*qubits[0], &mat);
            }
            U3 => {
                let theta = parameters[0] as f32;
                let phi = parameters[1] as f32;
                let lambda = parameters[2] as f32;
                let i = Complex32::i();
                let mat = [
                    (theta / 2.0).cos().into(),
                    -((i * lambda).exp()) * (theta / 2.0).sin(),
                    (i * phi).exp() * (theta / 2.0).sin(),
                    (i * (phi + lambda)).exp() * (theta / 2.0).cos(),
                ];
                self.0.u3(*qubits[0], &mat)
            }
            _ => {
                panic!("Unsupported quantum operation: {:?}", op_type);
            }
        }
        //self.0.dump();
    }
    fn measure(&mut self, qubit: &Self::Qubit)->bool{
        //self.0.dump();
        let (result, prob_zero) = self.0.measure_reset(*qubit, false, rand::random::<f32>());
        let prob_one = 1.0-prob_zero;
        trace!(
            "Measuring qubit {} (prob = ({}, {})), yielding {}",
            qubit,
            prob_zero,
            prob_one,
            result
        );

        match result {
            true => self.0.measure_res += "1",
            false => self.0.measure_res += "0"
        };
        //self.0.dump();
        result
    }

    fn get_measure_res(&mut self) -> String {
        return self.0.measure_res.clone();
    }
}