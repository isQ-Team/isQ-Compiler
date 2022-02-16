use super::super::super::context::get_current_context as context;
use super::types::*;
use crate::devices::qdevice::QuantumOp::*;
use crate::qdevice::QuantumOp;
use core::cell::RefCell;
pub fn isq_qir_shim_qis_u3(x0: K<QIRQubit>, theta: f64, phi: f64, lam: f64)->() {
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
        "calling isq_qir_shim_qis_h_ctl(x0: {})",
        x0
    );
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    let device = ctx.get_device_mut();
    device.controlled_qop(GPhase, &[], &[], &[x0]);
}
