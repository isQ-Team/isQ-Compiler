use super::impls::*;
use super::types::*;
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__u3(x0: *mut i8, x1: f64, x2: f64, x3: f64)->() {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_u3(t::<_, K<QIRQubit>>(x0), t::<_, f64>(x1), t::<_, f64>(x2), t::<_, f64>(x3)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__gphase(x0: f64)->() {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_gphase(t::<_, f64>(x0)))}
}
