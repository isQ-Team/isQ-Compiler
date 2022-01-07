use super::impls::*;
use super::types::*;
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__exp__body(x0: *mut i8, x1: f64, x2: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_exp_body(
            t::<_, K<QIRArray>>(x0),
            t::<_, f64>(x1),
            t::<_, K<QIRArray>>(x2),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__exp__adj(x0: *mut i8, x1: f64, x2: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_exp_adj(
            t::<_, K<QIRArray>>(x0),
            t::<_, f64>(x1),
            t::<_, K<QIRArray>>(x2),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__exp__ctl(
    x0: *mut i8,
    x1: *mut i8,
    x2: f64,
    x3: *mut i8,
) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_exp_ctl(
            t::<_, K<QIRArray>>(x0),
            t::<_, K<QIRArray>>(x1),
            t::<_, f64>(x2),
            t::<_, K<QIRArray>>(x3),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__exp__ctladj(
    x0: *mut i8,
    x1: *mut i8,
    x2: f64,
    x3: *mut i8,
) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_exp_ctladj(
            t::<_, K<QIRArray>>(x0),
            t::<_, K<QIRArray>>(x1),
            t::<_, f64>(x2),
            t::<_, K<QIRArray>>(x3),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__h__body(x0: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_h_body(t::<_, K<QIRQubit>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__h__ctl(x0: *mut i8, x1: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_h_ctl(
            t::<_, K<QIRArray>>(x0),
            t::<_, K<QIRQubit>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__measure__body(x0: *mut i8, x1: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_measure_body(
            t::<_, K<QIRArray>>(x0),
            t::<_, K<QIRArray>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__r__body(x0: i8, x1: f64, x2: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_r_body(
            t::<_, QIRPauli>(x0),
            t::<_, f64>(x1),
            t::<_, K<QIRQubit>>(x2),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__r__adj(x0: i8, x1: f64, x2: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_r_adj(
            t::<_, QIRPauli>(x0),
            t::<_, f64>(x1),
            t::<_, K<QIRQubit>>(x2),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__r__ctl(x0: *mut i8, x1: i8, x2: f64, x3: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_r_ctl(
            t::<_, K<QIRArray>>(x0),
            t::<_, QIRPauli>(x1),
            t::<_, f64>(x2),
            t::<_, K<QIRQubit>>(x3),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__r__ctladj(
    x0: *mut i8,
    x1: i8,
    x2: f64,
    x3: *mut i8,
) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_r_ctladj(
            t::<_, K<QIRArray>>(x0),
            t::<_, QIRPauli>(x1),
            t::<_, f64>(x2),
            t::<_, K<QIRQubit>>(x3),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__s__body(x0: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_s_body(t::<_, K<QIRQubit>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__s__adj(x0: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_s_adj(t::<_, K<QIRQubit>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__s__ctl(x0: *mut i8, x1: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_s_ctl(
            t::<_, K<QIRArray>>(x0),
            t::<_, K<QIRQubit>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__s__ctladj(x0: *mut i8, x1: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_s_ctladj(
            t::<_, K<QIRArray>>(x0),
            t::<_, K<QIRQubit>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__t__body(x0: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_t_body(t::<_, K<QIRQubit>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__t__adj(x0: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_t_adj(t::<_, K<QIRQubit>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__t__ctl(x0: *mut i8, x1: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_t_ctl(
            t::<_, K<QIRArray>>(x0),
            t::<_, K<QIRQubit>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__t__ctladj(x0: *mut i8, x1: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_t_ctladj(
            t::<_, K<QIRArray>>(x0),
            t::<_, K<QIRQubit>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__x__body(x0: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_x_body(t::<_, K<QIRQubit>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__x__ctl(x0: *mut i8, x1: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_x_ctl(
            t::<_, K<QIRArray>>(x0),
            t::<_, K<QIRQubit>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__y__body(x0: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_y_body(t::<_, K<QIRQubit>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__y__ctl(x0: *mut i8, x1: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_y_ctl(
            t::<_, K<QIRArray>>(x0),
            t::<_, K<QIRQubit>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__z__body(x0: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_z_body(t::<_, K<QIRQubit>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__z__ctl(x0: *mut i8, x1: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_z_ctl(
            t::<_, K<QIRArray>>(x0),
            t::<_, K<QIRQubit>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__dumpmachine__body(x0: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_dumpmachine_body(t::<_, *mut i8>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__dumpregister__body(x0: *mut i8, x1: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_qis_dumpregister_body(
            t::<_, *mut i8>(x0),
            t::<_, K<QIRArray>>(x1),
        ))
    }
}
