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
    trace!("calling isq_qir_shim_qis_cnot(x0: {})", P(&x0));
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    let device = ctx.get_device_mut();
    device.controlled_qop(CNOT, &[], &[&x0.key, &x1.key], &[]);
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
    println!("{}", x0);
    info!("{}", x0);
}
pub fn isq_qir_shim_qis_isq_print_f64(x0: f64)->() {
    extern crate std;
    use std::println;
    println!("{}", x0);
    info!("{}", x0);
}
