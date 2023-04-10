<<<<<<< HEAD
=======
use core::cell::RefCell;

>>>>>>> merge
use super::super::super::context::get_current_context as context;
use super::types::*;
use crate::devices::qdevice::QuantumOp::*;
use crate::qdevice::QuantumOp;
<<<<<<< HEAD
=======
use alloc::borrow::ToOwned;
>>>>>>> merge
use alloc::vec::Vec;
use itertools::Itertools;

pub fn isq_qir_shim_qis_exp_body(x0: K<QIRArray>, x1: f64, x2: K<QIRArray>) -> () {
    trace!(
        "calling isq_qir_shim_qis_exp_body(x0: {}, x1: {}, x2: {})",
        P(&x0),
        x1,
        P(&x2)
    );
    todo!()
}
pub fn isq_qir_shim_qis_exp_adj(x0: K<QIRArray>, x1: f64, x2: K<QIRArray>) -> () {
    trace!(
        "calling isq_qir_shim_qis_exp_adj(x0: {}, x1: {}, x2: {})",
        P(&x0),
        x1,
        P(&x2)
    );
    todo!()
}
pub fn isq_qir_shim_qis_exp_ctl(x0: K<QIRArray>, x1: K<QIRArray>, x2: f64, x3: K<QIRArray>) -> () {
    trace!(
        "calling isq_qir_shim_qis_exp_ctl(x0: {}, x1: {}, x2: {}, x3: {})",
        P(&x0),
        P(&x1),
        x2,
        P(&x3)
    );
    todo!()
}
pub fn isq_qir_shim_qis_exp_ctladj(
    x0: K<QIRArray>,
    x1: K<QIRArray>,
    x2: f64,
    x3: K<QIRArray>,
) -> () {
    trace!(
        "calling isq_qir_shim_qis_exp_ctladj(x0: {}, x1: {}, x2: {}, x3: {})",
        P(&x0),
        P(&x1),
        x2,
        P(&x3)
    );
    todo!()
}
pub fn isq_qir_shim_qis_h_body(x0: K<QIRQubit>) -> () {
    trace!("calling isq_qir_shim_qis_h_body(x0: {})", P(&x0));
    let rctx = context();
<<<<<<< HEAD
    let mut ctx = rctx.lock().unwrap();
=======
    let mut ctx = RefCell::borrow_mut(&rctx);
>>>>>>> merge
    let device = ctx.get_device_mut();
    device.controlled_qop(H, &[], &[&x0.key], &[]);
}
pub fn isq_qir_shim_qis_h_ctl(x0: K<QIRArray>, x1: K<QIRQubit>) -> () {
    trace!(
        "calling isq_qir_shim_qis_h_ctl(x0: {}, x1: {})",
        P(&x0),
        P(&x1)
    );
    let rctx = context();
<<<<<<< HEAD
    let mut ctx = rctx.lock().unwrap();
=======
    let mut ctx = RefCell::borrow_mut(&rctx);
>>>>>>> merge
    let controls: Vec<usize> = x0
        .get(&ctx)
        .get_1d_data_of::<usize>()
        .iter()
        .copied()
        .collect_vec();
    let controls_refs = controls.iter().collect_vec();
    let device = ctx.get_device_mut();
    device.controlled_qop(H, &controls_refs, &[&x1.key], &[]);
}
pub fn isq_qir_shim_qis_measure_body(x0: K<QIRArray>, x1: K<QIRArray>) -> QIRResult {
    trace!(
        "calling isq_qir_shim_qis_measure_body(x0: {}, x1: {})",
        P(&x0),
        P(&x1)
    );
    let rctx = context();
<<<<<<< HEAD
    let mut ctx = rctx.lock().unwrap();
=======
    let mut ctx = RefCell::borrow_mut(&rctx);
>>>>>>> merge
    let paulis = x0
        .get(&ctx)
        .get_1d_data_of::<QIRPauli>()
        .iter()
        .copied()
        .collect_vec();
    let qubits = x1
        .get(&ctx)
        .get_1d_data_of::<usize>()
        .iter()
        .copied()
        .collect_vec();
    if paulis.len() != qubits.len() {
        panic!("paulis and qubits must be of the same length");
    }
    let device = ctx.get_device_mut();
    let mut ret = false;
    for (p, q) in paulis.iter().zip(qubits.iter()) {
        let mut tmp = false;
        match p {
            QIRPauli::PauliI => {
                // do nothing.
                warn!("Trying to measure qubit {} with I. Do nothing.", q);
            }
            QIRPauli::PauliX => {
                device.qop(H, &[&q], &[]);
                tmp = device.measure(&q);
                device.qop(H, &[&q], &[]);
            }
            QIRPauli::PauliY => {
                device.qop(SInv, &[&q], &[]);
                device.qop(H, &[&q], &[]);
                tmp = device.measure(&q);
                device.qop(H, &[&q], &[]);
                device.qop(S, &[&q], &[]);
            }
            QIRPauli::PauliZ => {
                tmp = device.measure(&q);
            }
        }
        ret ^= tmp;
    }
    if ret {
        QIR_RESULT_ONE
    } else {
        QIR_RESULT_ZERO
    }
}
fn pauli_to_qop(qop: QIRPauli) -> Option<QuantumOp> {
    match qop {
        QIRPauli::PauliX => Some(Rx),
        QIRPauli::PauliY => Some(Ry),
        QIRPauli::PauliZ => Some(Rz),
        QIRPauli::PauliI => None,
    }
}
use super::super::sq_op;
pub fn isq_qir_shim_qis_r_body(x0: QIRPauli, x1: f64, x2: K<QIRQubit>) -> () {
    trace!(
        "calling isq_qir_shim_qis_r_body(x0: {:?}, x1: {}, x2: {})",
        x0,
        x1,
        P(&x2)
    );
    if let Some(pauli) = pauli_to_qop(x0) {
        sq_op(None, pauli, &[x1], x2.key);
    }
}
pub fn isq_qir_shim_qis_r_adj(x0: QIRPauli, x1: f64, x2: K<QIRQubit>) -> () {
    trace!(
        "calling isq_qir_shim_qis_r_adj(x0: {:?}, x1: {}, x2: {})",
        x0,
        x1,
        P(&x2)
    );
    if let Some(pauli) = pauli_to_qop(x0) {
        sq_op(None, pauli, &[-x1], x2.key);
    }
}
pub fn isq_qir_shim_qis_r_ctl(x0: K<QIRArray>, x1: QIRPauli, x2: f64, x3: K<QIRQubit>) -> () {
    trace!(
        "calling isq_qir_shim_qis_r_ctl(x0: {}, x1: {:?}, x2: {}, x3: {})",
        P(&x0),
        x1,
        x2,
        P(&x3)
    );
    if let Some(pauli) = pauli_to_qop(x1) {
        sq_op(Some(x0), pauli, &[x2], x3.key);
    }
}
pub fn isq_qir_shim_qis_r_ctladj(x0: K<QIRArray>, x1: QIRPauli, x2: f64, x3: K<QIRQubit>) -> () {
    trace!(
        "calling isq_qir_shim_qis_r_ctladj(x0: {}, x1: {:?}, x2: {}, x3: {})",
        P(&x0),
        x1,
        x2,
        P(&x3)
    );
    if let Some(pauli) = pauli_to_qop(x1) {
        sq_op(Some(x0), pauli, &[-x2], x3.key);
    }
}
pub fn isq_qir_shim_qis_s_body(x0: K<QIRQubit>) -> () {
    trace!("calling isq_qir_shim_qis_s_body(x0: {})", P(&x0));
    sq_op(None, S, &[], x0.key)
}
pub fn isq_qir_shim_qis_s_adj(x0: K<QIRQubit>) -> () {
    trace!("calling isq_qir_shim_qis_s_adj(x0: {})", P(&x0));
    sq_op(None, SInv, &[], x0.key)
}
pub fn isq_qir_shim_qis_s_ctl(x0: K<QIRArray>, x1: K<QIRQubit>) -> () {
    trace!(
        "calling isq_qir_shim_qis_s_ctl(x0: {}, x1: {})",
        P(&x0),
        P(&x1)
    );
    sq_op(Some(x0), S, &[], x1.key)
}
pub fn isq_qir_shim_qis_s_ctladj(x0: K<QIRArray>, x1: K<QIRQubit>) -> () {
    trace!(
        "calling isq_qir_shim_qis_s_ctladj(x0: {}, x1: {})",
        P(&x0),
        P(&x1)
    );
    sq_op(Some(x0), SInv, &[], x1.key)
}
pub fn isq_qir_shim_qis_t_body(x0: K<QIRQubit>) -> () {
    trace!("calling isq_qir_shim_qis_t_body(x0: {})", P(&x0));
    sq_op(None, T, &[], x0.key)
}
pub fn isq_qir_shim_qis_t_adj(x0: K<QIRQubit>) -> () {
    trace!("calling isq_qir_shim_qis_t_adj(x0: {})", P(&x0));
    sq_op(None, TInv, &[], x0.key)
}
pub fn isq_qir_shim_qis_t_ctl(x0: K<QIRArray>, x1: K<QIRQubit>) -> () {
    trace!(
        "calling isq_qir_shim_qis_t_ctl(x0: {}, x1: {})",
        P(&x0),
        P(&x1)
    );
    sq_op(Some(x0), T, &[], x1.key)
}
pub fn isq_qir_shim_qis_t_ctladj(x0: K<QIRArray>, x1: K<QIRQubit>) -> () {
    trace!(
        "calling isq_qir_shim_qis_t_ctladj(x0: {}, x1: {})",
        P(&x0),
        P(&x1)
    );
    sq_op(Some(x0), TInv, &[], x1.key)
}
pub fn isq_qir_shim_qis_x_body(x0: K<QIRQubit>) -> () {
    trace!("calling isq_qir_shim_qis_x_body(x0: {})", P(&x0));
    sq_op(None, X, &[], x0.key)
}
pub fn isq_qir_shim_qis_x_ctl(x0: K<QIRArray>, x1: K<QIRQubit>) -> () {
    trace!(
        "calling isq_qir_shim_qis_x_ctl(x0: {}, x1: {})",
        P(&x0),
        P(&x1)
    );
    sq_op(Some(x0), X, &[], x1.key)
}
pub fn isq_qir_shim_qis_y_body(x0: K<QIRQubit>) -> () {
    trace!("calling isq_qir_shim_qis_y_body(x0: {})", P(&x0));
    sq_op(None, Y, &[], x0.key)
}
pub fn isq_qir_shim_qis_y_ctl(x0: K<QIRArray>, x1: K<QIRQubit>) -> () {
    trace!(
        "calling isq_qir_shim_qis_y_ctl(x0: {}, x1: {})",
        P(&x0),
        P(&x1)
    );
    sq_op(Some(x0), Y, &[], x1.key)
}
pub fn isq_qir_shim_qis_z_body(x0: K<QIRQubit>) -> () {
    trace!("calling isq_qir_shim_qis_z_body(x0: {})", P(&x0));
    sq_op(None, Z, &[], x0.key)
}
pub fn isq_qir_shim_qis_z_ctl(x0: K<QIRArray>, x1: K<QIRQubit>) -> () {
    trace!(
        "calling isq_qir_shim_qis_z_ctl(x0: {}, x1: {})",
        P(&x0),
        P(&x1)
    );
    sq_op(Some(x0), Z, &[], x1.key)
}
pub fn isq_qir_shim_qis_dumpmachine_body(x0: *mut i8) -> () {
    trace!("calling isq_qir_shim_qis_dumpmachine_body(x0: {:?})", x0);
    let rctx = context();
<<<<<<< HEAD
    let mut ctx = rctx.lock().unwrap();
=======
    let mut ctx = RefCell::borrow_mut(&rctx);
>>>>>>> merge
    ctx.dump_machine();
}
pub fn isq_qir_shim_qis_dumpregister_body(x0: *mut i8, x1: K<QIRArray>) -> () {
    trace!(
        "calling isq_qir_shim_qis_dumpregister_body(x0: {:?}, x1: {})",
        x0,
        P(&x1)
    );
    let rctx = context();
<<<<<<< HEAD
    let mut ctx = rctx.lock().unwrap();
=======
    let mut ctx = RefCell::borrow_mut(&rctx);
>>>>>>> merge
    ctx.dump_registers(x1);
}
