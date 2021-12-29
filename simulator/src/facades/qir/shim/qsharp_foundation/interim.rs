use super::impls::*;
use super::types::*;
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__nan__body()->f64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_nan_body())}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__isnan__body(x0: f64)->bool {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_isnan_body(t::<_, f64>(x0)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__infinity__body()->f64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_infinity_body())}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__isinf__body(x0: f64)->bool {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_isinf_body(t::<_, f64>(x0)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__isnegativeinfinity__body(x0: f64)->bool {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_isnegativeinfinity_body(t::<_, f64>(x0)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__sin__body(x0: f64)->f64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_sin_body(t::<_, f64>(x0)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__cos__body(x0: f64)->f64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_cos_body(t::<_, f64>(x0)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__tan__body(x0: f64)->f64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_tan_body(t::<_, f64>(x0)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__arctan2__body(x0: f64, x1: f64)->f64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_arctan2_body(t::<_, f64>(x0), t::<_, f64>(x1)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__sinh__body(x0: f64)->f64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_sinh_body(t::<_, f64>(x0)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__cosh__body(x0: f64)->f64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_cosh_body(t::<_, f64>(x0)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__tanh__body(x0: f64)->f64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_tanh_body(t::<_, f64>(x0)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__arcsin__body(x0: f64)->f64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_arcsin_body(t::<_, f64>(x0)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__arccos__body(x0: f64)->f64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_arccos_body(t::<_, f64>(x0)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__arctan__body(x0: f64)->f64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_arctan_body(t::<_, f64>(x0)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__sqrt__body(x0: f64)->f64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_sqrt_body(t::<_, f64>(x0)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__log__body(x0: f64)->f64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_log_body(t::<_, f64>(x0)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__ieeeremainder__body(x0: f64, x1: f64)->f64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_ieeeremainder_body(t::<_, f64>(x0), t::<_, f64>(x1)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__drawrandomint__body(x0: i64, x1: i64)->i64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_drawrandomint_body(t::<_, i64>(x0), t::<_, i64>(x1)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__drawrandomdouble__body(x0: f64, x1: f64)->f64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_drawrandomdouble_body(t::<_, f64>(x0), t::<_, f64>(x1)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__applyifelseintrinsic__body(x0: *mut i8, x1: *mut i8, x2: *mut i8)->() {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_applyifelseintrinsic_body(t::<_, QIRResult>(x0), t::<_, K<QIRCallable>>(x1), t::<_, K<QIRCallable>>(x2)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__applyconditionallyinstrinsic__body(x0: *mut i8, x1: *mut i8, x2: *mut i8, x3: *mut i8)->() {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_applyconditionallyinstrinsic_body(t::<_, K<QIRArray>>(x0), t::<_, K<QIRArray>>(x1), t::<_, K<QIRCallable>>(x2), t::<_, K<QIRCallable>>(x3)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__assertmeasurementprobability__body(x0: *mut i8, x1: *mut i8, x2: *mut i8, x3: f64, x4: *mut i8, x5: f64)->() {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_assertmeasurementprobability_body(t::<_, K<QIRArray>>(x0), t::<_, K<QIRArray>>(x1), t::<_, QIRResult>(x2), t::<_, f64>(x3), t::<_, K<QIRString>>(x4), t::<_, f64>(x5)))}
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__qis__assertmeasurementprobability__ctl(x0: *mut i8, x1: *mut i8)->() {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_qis_assertmeasurementprobability_ctl(t::<_, K<QIRArray>>(x0), t::<_, *const MeasurementProbabilityArgs>(x1)))}
}
