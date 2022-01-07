use super::impls::*;
use super::types::*;
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__array_concatenate(x0: *mut i8, x1: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_array_concatenate(
            t::<_, K<QIRArray>>(x0),
            t::<_, K<QIRArray>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__array_copy(x0: *mut i8, x1: bool) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_array_copy(
            t::<_, K<QIRArray>>(x0),
            t::<_, bool>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__array_create(x0: i32, x1: i32, x2: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_array_create(
            t::<_, i32>(x0),
            t::<_, i32>(x1),
            t::<_, *mut i64>(x2),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__array_create_1d(x0: i32, x1: i64) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_array_create_1d(
            t::<_, i32>(x0),
            t::<_, i64>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__array_get_dim(x0: *mut i8) -> i32 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_array_get_dim(t::<_, K<QIRArray>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__array_get_element_ptr(x0: *mut i8, x1: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_array_get_element_ptr(
            t::<_, K<QIRArray>>(x0),
            t::<_, *mut i64>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__array_get_element_ptr_1d(x0: *mut i8, x1: i64) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_array_get_element_ptr_1d(
            t::<_, K<QIRArray>>(x0),
            t::<_, i64>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__array_get_size(x0: *mut i8, x1: i32) -> i64 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_array_get_size(
            t::<_, K<QIRArray>>(x0),
            t::<_, i32>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__array_get_size_1d(x0: *mut i8) -> i64 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_array_get_size_1d(t::<_, K<QIRArray>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__array_project(
    x0: *mut i8,
    x1: i32,
    x2: i64,
    x3: bool,
) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_array_project(
            t::<_, K<QIRArray>>(x0),
            t::<_, i32>(x1),
            t::<_, i64>(x2),
            t::<_, bool>(x3),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__array_slice(
    x0: *mut i8,
    x1: i32,
    x2: *mut i8,
    x3: bool,
) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_array_slice(
            t::<_, K<QIRArray>>(x0),
            t::<_, i32>(x1),
            t::<_, *const QIRRange>(x2),
            t::<_, bool>(x3),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__array_slice_1d(
    x0: *mut i8,
    x1: *mut i8,
    x2: bool,
) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_array_slice_1d(
            t::<_, K<QIRArray>>(x0),
            t::<_, *const QIRRange>(x1),
            t::<_, bool>(x2),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__array_update_alias_count(x0: *mut i8, x1: i32) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_array_update_alias_count(
            t::<_, K<QIRArray>>(x0),
            t::<_, i32>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__array_update_reference_count(x0: *mut i8, x1: i32) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_array_update_reference_count(
            t::<_, K<QIRArray>>(x0),
            t::<_, i32>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_add(x0: *mut i8, x1: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_bigint_add(
            t::<_, K<QIRBigInt>>(x0),
            t::<_, K<QIRBigInt>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_bitand(x0: *mut i8, x1: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_bigint_bitand(
            t::<_, K<QIRBigInt>>(x0),
            t::<_, K<QIRBigInt>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_bitor(x0: *mut i8, x1: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_bigint_bitor(
            t::<_, K<QIRBigInt>>(x0),
            t::<_, K<QIRBigInt>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_bitxor(x0: *mut i8, x1: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_bigint_bitxor(
            t::<_, K<QIRBigInt>>(x0),
            t::<_, K<QIRBigInt>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_create_array(x0: i32, x1: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_bigint_create_array(
            t::<_, i32>(x0),
            t::<_, *mut i8>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_create_i64(x0: i64) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_bigint_create_i64(t::<_, i64>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_divide(x0: *mut i8, x1: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_bigint_divide(
            t::<_, K<QIRBigInt>>(x0),
            t::<_, K<QIRBigInt>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_equal(x0: *mut i8, x1: *mut i8) -> bool {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_bigint_equal(
            t::<_, K<QIRBigInt>>(x0),
            t::<_, K<QIRBigInt>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_get_data(x0: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_bigint_get_data(t::<_, K<QIRBigInt>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_get_length(x0: *mut i8) -> i32 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_bigint_get_length(t::<_, K<QIRBigInt>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_greater(x0: *mut i8, x1: *mut i8) -> bool {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_bigint_greater(
            t::<_, K<QIRBigInt>>(x0),
            t::<_, K<QIRBigInt>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_greater_eq(x0: *mut i8, x1: *mut i8) -> bool {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_bigint_greater_eq(
            t::<_, K<QIRBigInt>>(x0),
            t::<_, K<QIRBigInt>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_modulus(x0: *mut i8, x1: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_bigint_modulus(
            t::<_, K<QIRBigInt>>(x0),
            t::<_, K<QIRBigInt>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_multiply(x0: *mut i8, x1: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_bigint_multiply(
            t::<_, K<QIRBigInt>>(x0),
            t::<_, K<QIRBigInt>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_negate(x0: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_bigint_negate(t::<_, K<QIRBigInt>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_power(x0: *mut i8, x1: i32) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_bigint_power(
            t::<_, K<QIRBigInt>>(x0),
            t::<_, i32>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_shiftleft(x0: *mut i8, x1: i64) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_bigint_shiftleft(
            t::<_, K<QIRBigInt>>(x0),
            t::<_, i64>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_shiftright(x0: *mut i8, x1: i64) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_bigint_shiftright(
            t::<_, K<QIRBigInt>>(x0),
            t::<_, i64>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_subtract(x0: *mut i8, x1: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_bigint_subtract(
            t::<_, K<QIRBigInt>>(x0),
            t::<_, K<QIRBigInt>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_to_string(x0: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_bigint_to_string(t::<_, K<QIRBigInt>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bigint_update_reference_count(x0: *mut i8, x1: i32) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_bigint_update_reference_count(
            t::<_, K<QIRBigInt>>(x0),
            t::<_, i32>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__bool_to_string(x0: bool) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_bool_to_string(t::<_, bool>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__callable_copy(x0: *mut i8, x1: bool) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_callable_copy(
            t::<_, K<QIRCallable>>(x0),
            t::<_, bool>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__callable_create(
    x0: *mut i8,
    x1: *mut i8,
    x2: *mut i8,
    x3: *mut i8,
    x4: *mut i8,
    x5: *mut i8,
    x6: *mut i8,
) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_callable_create(
            t::<_, *mut i8>(x0),
            t::<_, *mut i8>(x1),
            t::<_, *mut i8>(x2),
            t::<_, *mut i8>(x3),
            t::<_, *mut i8>(x4),
            t::<_, *mut i8>(x5),
            t::<_, TupleBodyPtr>(x6),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__callable_invoke(
    x0: *mut i8,
    x1: *mut i8,
    x2: *mut i8,
) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_callable_invoke(
            t::<_, K<QIRCallable>>(x0),
            t::<_, TupleBodyPtr>(x1),
            t::<_, TupleBodyPtr>(x2),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__callable_make_adjoint(x0: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_callable_make_adjoint(
            t::<_, K<QIRCallable>>(x0),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__callable_make_controlled(x0: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_callable_make_controlled(t::<
            _,
            K<QIRCallable>,
        >(x0)))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__callable_update_alias_count(x0: *mut i8, x1: i32) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_callable_update_alias_count(
            t::<_, K<QIRCallable>>(x0),
            t::<_, i32>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__callable_update_reference_count(
    x0: *mut i8,
    x1: i32,
) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_callable_update_reference_count(
            t::<_, K<QIRCallable>>(x0),
            t::<_, i32>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__capture_update_alias_count(x0: *mut i8, x1: i32) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_capture_update_alias_count(
            t::<_, K<QIRCallable>>(x0),
            t::<_, i32>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__capture_update_reference_count(x0: *mut i8, x1: i32) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_capture_update_reference_count(
            t::<_, K<QIRCallable>>(x0),
            t::<_, i32>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__double_to_string(x0: f64) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_double_to_string(t::<_, f64>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__fail(x0: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_fail(t::<_, K<QIRString>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__int_to_string(x0: i64) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_int_to_string(t::<_, i64>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__message(x0: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_message(t::<_, K<QIRString>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__pauli_to_string(x0: i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_pauli_to_string(t::<_, QIRPauli>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__qubit_allocate() -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_qubit_allocate()) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__qubit_allocate_array(x0: i32) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_qubit_allocate_array(t::<_, i32>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__qubit_release(x0: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_qubit_release(t::<_, K<QIRQubit>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__qubit_release_array(x0: *mut i8) -> () {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_qubit_release_array(t::<_, K<QIRArray>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__qubit_to_string(x0: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_qubit_to_string(t::<_, K<QIRQubit>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__range_to_string(x0: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_range_to_string(t::<_, *const QIRRange>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__result_equal(x0: *mut i8, x1: *mut i8) -> bool {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_result_equal(
            t::<_, QIRResult>(x0),
            t::<_, QIRResult>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__result_get_one() -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_result_get_one()) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__result_get_zero() -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_result_get_zero()) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__result_to_string(x0: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_result_to_string(t::<_, QIRResult>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__result_update_reference_count(x0: *mut i8, x1: i32) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_result_update_reference_count(
            t::<_, QIRResult>(x0),
            t::<_, i32>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__string_concatenate(x0: *mut i8, x1: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_string_concatenate(
            t::<_, K<QIRString>>(x0),
            t::<_, K<QIRString>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__string_create(x0: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_string_create(t::<_, *mut i8>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__string_equal(x0: *mut i8, x1: *mut i8) -> bool {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_string_equal(
            t::<_, K<QIRString>>(x0),
            t::<_, K<QIRString>>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__string_get_data(x0: *mut i8) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_string_get_data(t::<_, K<QIRString>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__string_get_length(x0: *mut i8) -> i32 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_string_get_length(t::<_, K<QIRString>>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__string_update_reference_count(x0: *mut i8, x1: i32) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_string_update_reference_count(
            t::<_, K<QIRString>>(x0),
            t::<_, i32>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__tuple_copy(x0: *mut i8, x1: bool) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_tuple_copy(
            t::<_, TupleBodyPtr>(x0),
            t::<_, bool>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__tuple_create(x0: i64) -> *mut i8 {
    use core::mem::transmute as t;
    unsafe { t(isq_qir_shim_rt_tuple_create(t::<_, i64>(x0))) }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__tuple_update_alias_count(x0: *mut i8, x1: i32) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_tuple_update_alias_count(
            t::<_, TupleBodyPtr>(x0),
            t::<_, i32>(x1),
        ))
    }
}
#[no_mangle]
pub extern "C" fn __isq__qir__shim__rt__tuple_update_reference_count(x0: *mut i8, x1: i32) -> () {
    use core::mem::transmute as t;
    unsafe {
        t(isq_qir_shim_rt_tuple_update_reference_count(
            t::<_, TupleBodyPtr>(x0),
            t::<_, i32>(x1),
        ))
    }
}
