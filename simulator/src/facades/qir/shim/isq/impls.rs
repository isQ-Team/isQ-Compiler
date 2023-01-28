use super::super::super::context::get_current_context as context;
use super::types::*;
use crate::devices::qdevice::QuantumOp::*;
use crate::qdevice::QuantumOp;
use core::cell::RefCell;
pub fn isq_qir_shim_qis_u3(theta: f64, phi: f64, lam: f64, x0: K<QIRQubit>)->() {
    trace!(
        "calling isq_qir_shim_qis_u3(x0: {}, theta: {}, phi: {}, lam: {})",
        P(&x0),
        theta, phi, lam
    );
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    let device = ctx.get_device_mut();
    device.controlled_qop(U3, &[], &[&x0.key], &[theta, phi, lam]);
}
pub fn isq_qir_shim_qis_rz_body(x0: f64, x1: K<QIRQubit>)->() {
    trace!(
        "calling isq_qir_shim_qis_rz(x0: {}, theta: {})",
        P(&x1),
        x0,
    );
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    let device = ctx.get_device_mut();
    device.controlled_qop(Rz, &[], &[&x1.key], &[x0]);
}
pub fn isq_qir_shim_qis_x2p(x0: K<QIRQubit>)->() {
    trace!(
        "calling isq_qir_shim_qis_x2p(x0: {})",
        P(&x0)
    );
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    let device = ctx.get_device_mut();
    device.controlled_qop(X2P, &[], &[&x0.key], &[]);
}
pub fn isq_qir_shim_qis_x2m(x0: K<QIRQubit>)->() {
    trace!(
        "calling isq_qir_shim_qis_x2m(x0: {})",
        P(&x0)
    );
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    let device = ctx.get_device_mut();
    device.controlled_qop(X2M, &[], &[&x0.key], &[]);
}
pub fn isq_qir_shim_qis_y2p(x0: K<QIRQubit>)->() {
    trace!(
        "calling isq_qir_shim_qis_y2y(x0: {})",
        P(&x0)
    );
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    let device = ctx.get_device_mut();
    device.controlled_qop(Y2P, &[], &[&x0.key], &[]);
}
pub fn isq_qir_shim_qis_y2m(x0: K<QIRQubit>)->() {
    trace!(
        "calling isq_qir_shim_qis_y2m(x0: {})",
        P(&x0)
    );
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    let device = ctx.get_device_mut();
    device.controlled_qop(Y2M, &[], &[&x0.key], &[]);
}

pub fn isq_qir_shim_qis_gphase(x0: f64)->() {
    trace!(
        "calling isq_qir_shim_qis_gphase(x0: {})",
        x0
    );
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    let device = ctx.get_device_mut();
    device.controlled_qop(GPhase, &[], &[], &[x0]);
}
pub fn isq_qir_shim_qis_cnot(x0: K<QIRQubit>, x1: K<QIRQubit>)->() {
    trace!("calling isq_qir_shim_qis_cnot(x0: {}, x1: {})", P(&x0), P(&x1));
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    let device = ctx.get_device_mut();
    device.controlled_qop(CNOT, &[], &[&x0.key, &x1.key], &[]);
}
pub fn isq_qir_shim_qis_cz(x0: K<QIRQubit>, x1: K<QIRQubit>)->() {
    trace!("calling isq_qir_shim_qis_cz(x0: {}, x1: {})", P(&x0), P(&x1));
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    let device = ctx.get_device_mut();
    device.controlled_qop(CZ, &[], &[&x0.key, &x1.key], &[]);
}
pub fn isq_qir_shim_qis_measure(x0: K<QIRQubit>)->QIRResult {
    trace!("calling isq_qir_shim_qis_measure(x0: {})", P(&x0));
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    let device = ctx.get_device_mut();
    if device.measure(&x0.key){
        QIR_RESULT_ONE
    }else{
        QIR_RESULT_ZERO
    }

}
pub fn isq_qir_shim_qis_reset(x0: K<QIRQubit>)->() {
    trace!("calling isq_qir_shim_qis_reset(x0: {})", P(&x0));
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    let device = ctx.get_device_mut();
    device.controlled_qop(Reset, &[], &[&x0.key], &[]);
}


pub fn isq_qir_shim_qis_isq_print_i64(x0: i64)->() {
    extern crate std;
    use std::println;
    //println!("{}", x0);
    debug!("{}", x0);
}
pub fn isq_qir_shim_qis_isq_print_f64(x0: f64)->() {
    extern crate std;
    use std::println;
    //println!("{}", x0);
    debug!("{}", x0);
}
pub fn isq_qir_shim_qis_qcis_finalize()->() {
    trace!(
        "calling isq_qir_shim_qis_qcis_finalize()",
    );
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    let device = ctx.get_device_mut();
    device.controlled_qop(QCIS_Finalize, &[], &[], &[]);
}
pub fn isq_qir_shim_qis_bp(x0: i64)->() {
    trace!(
        "calling isq_qir_shim_qis_bp()",
    );
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    if ctx.contains_bp(x0) {
        return;
    }
    extern crate std;
    use std::println;
    use std::string::String;
    println!("Reaching a break point");
    let device = ctx.get_device();
    device.print_state();
    let mut buffer = String::new();
    std::io::stdin().read_line(&mut buffer).ok().expect("Failed to read line");
    if buffer.trim() == "d" {
        ctx.disable_bp_index(x0);
    }
}
