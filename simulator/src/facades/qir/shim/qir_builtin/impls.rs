use core::borrow::Borrow;
use core::cell::Ref;
use alloc::vec::Vec;
use crate::facades::qir::callable::QCallable;
use crate::facades::qir::resource::{AliasingTracker, ResourceManager};

use super::super::super::context::get_current_context as context;
use super::types::*;

pub fn isq_qir_shim_rt_array_concatenate(x0: K<QIRArray>, x1: K<QIRArray>) -> K<QIRArray> {
    trace!(
        "calling qir_shim_rt_array_concatenate({}, {})",
        P(&x0),
        P(&x1)
    );
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = a0.concat_1d(a1.borrow());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_array_copy(x0: K<QIRArray>, x1: bool) -> K<QIRArray> {
    trace!("calling qir_shim_rt_array_copy({}, {})", P(&x0), x1);
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    x0.try_copy(&ctx, x1)
}
pub fn isq_qir_shim_rt_array_create(size: i32, num_dims: i32, pointer: *mut i64) -> K<QIRArray> {
    trace!("calling qir_shim_rt_array_create({}, {})", size, num_dims);
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let dimensions =
        unsafe { core::slice::from_raw_parts_mut(pointer as *mut usize, num_dims as usize) };
    let a = QIRArray::new(size as usize, dimensions);
    ctx.add(a)
}
pub fn isq_qir_shim_rt_array_create_1d(x0: i32, mut x1: i64) -> K<QIRArray> {
    trace!("calling qir_shim_rt_array_create_1d({}, {})", x0, x1);
    isq_qir_shim_rt_array_create(x0, 1, &mut x1 as *mut _)
}
pub fn isq_qir_shim_rt_array_get_dim(x0: K<QIRArray>) -> i32 {
    trace!("calling qir_shim_rt_array_get_dim({})", P(&x0));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let x = x0.get(&ctx).get_dimensions().len() as i32;
    x
}
pub fn isq_qir_shim_rt_array_get_element_ptr(x0: K<QIRArray>, x1: *mut i64) -> *mut i8 {
    trace!(
        "calling qir_shim_rt_array_get_element_ptr({}, {})",
        P(&x0),
        x1 as usize
    );
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let indices =
        unsafe { core::slice::from_raw_parts_mut(x1, x0.get(&ctx).get_dimensions().len()) };
    trace!("decoded indices: {:?}", indices);
    let x = x0.get(&ctx).get_element(indices);
    x
}
pub fn isq_qir_shim_rt_array_get_element_ptr_1d(x0: K<QIRArray>, x1: i64) -> *mut i8 {
    trace!(
        "calling qir_shim_rt_array_get_element_ptr_1d({}, {})",
        P(&x0),
        x1
    );
    trace!("forwarding to qir_shim_rt_array_get_element_ptr");
    let mut x2 = x1;
    isq_qir_shim_rt_array_get_element_ptr(x0, &mut x2 as *mut _)
}
pub fn isq_qir_shim_rt_array_get_size(x0: K<QIRArray>, x1: i32) -> i64 {
    trace!("calling qir_shim_rt_array_get_size({}, {})", P(&x0), x1);
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let x = x0.get(&ctx).get_dimensions()[x1 as usize] as i64;
    x
}
pub fn isq_qir_shim_rt_array_get_size_1d(x0: K<QIRArray>) -> i64 {
    trace!("calling qir_shim_rt_array_get_size_1d({})", P(&x0));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    if a0.get_dimensions().len() != 1 {
        panic!(
            "Array has {} dimensions, expected 1",
            a0.get_dimensions().len()
        );
    }
    a0.get_dimensions()[0] as i64
}
pub fn isq_qir_shim_rt_array_project(
    x0: K<QIRArray>,
    index: i32,
    id: i64,
    x3: bool,
) -> K<QIRArray> {
    trace!(
        "calling qir_shim_rt_array_project({}, {}, {}, {})",
        P(&x0),
        index,
        id,
        x3
    );
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a = a0.project(index as usize, id as usize);
    // special handling for array.
    if a0.get_alias_count() == 0 && !x3 {
        drop(a0);
        drop(ctx);
        let mut ctx = rctx.lock().unwrap();
        let mut a0 = x0.get_mut(&ctx);
        *a0 = a;
        x0.update_ref_count(&ctx, 1);
        x0
    } else {
        ctx.add(a)
    }
}
pub fn isq_qir_shim_rt_array_slice(
    x0: K<QIRArray>,
    x1: i32,
    start: i64,
    step: i64,
    end: i64,
    x3: bool,
) -> K<QIRArray> {
    let x2 = QIRRange{start, step, end};
    trace!(
        "calling qir_shim_rt_array_slice({}, {}, {:?}, {})",
        P(&x0),
        x1,
        &x2,
        x3
    );
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a = a0.slice(x1 as usize, x2);
    // special handling for array.
    if a0.get_alias_count() == 0 && !x3 {
        drop(a0);
        drop(ctx);
        let mut ctx = rctx.lock().unwrap();
        let mut a0 = x0.get_mut(&ctx);
        *a0 = a;
        x0.update_ref_count(&ctx, 1);
        x0
    } else {
        drop(a0);
        ctx.add(a)
    }
}
pub fn isq_qir_shim_rt_array_slice_1d(
    x0: K<QIRArray>,
    start: i64,
    step: i64,
    end: i64,
    x2: bool,
) -> K<QIRArray> {
    let x1 = QIRRange{start, step, end};
    trace!("{:?}", x1);
    trace!(
        "calling qir_shim_rt_array_slice_1d({}, {:?}, {})",
        P(&x0),
        x1,
        x2
    );
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    if a0.get_dimensions().len() != 1 {
        panic!(
            "Array has {} dimensions, expected 1",
            a0.get_dimensions().len()
        );
    }
    drop(a0);
    drop(ctx);
    drop(rctx);
    isq_qir_shim_rt_array_slice(x0, 0, x1.start, x1.step, x1.end, x2)
}
pub fn isq_qir_shim_rt_array_update_alias_count(x0: K<QIRArray>, x1: i32) -> () {
    trace!(
        "calling qir_shim_rt_array_update_alias_count({}, {})",
        P(&x0),
        x1
    );
    if x0.is_null() {
        return;
    }
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    x0.update_alias_count(&ctx, x1 as isize)
}
pub fn isq_qir_shim_rt_array_update_reference_count(x0: K<QIRArray>, x1: i32) -> () {
    trace!(
        "calling qir_shim_rt_array_update_reference_count({}, {})",
        P(&x0),
        x1
    );
    if x0.is_null() {
        return;
    }
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    x0.update_ref_count(&ctx, x1 as isize)
}
pub fn isq_qir_shim_rt_bigint_add(x0: K<QIRBigInt>, x1: K<QIRBigInt>) -> K<QIRBigInt> {
    trace!("calling qir_shim_rt_bigint_add({}, {})", P(&x0), P(&x1));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() + a1.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_bitand(x0: K<QIRBigInt>, x1: K<QIRBigInt>) -> K<QIRBigInt> {
    trace!("calling qir_shim_rt_bigint_bitand({}, {})", P(&x0), P(&x1));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() & a1.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_bitor(x0: K<QIRBigInt>, x1: K<QIRBigInt>) -> K<QIRBigInt> {
    trace!("calling qir_shim_rt_bigint_bitor({}, {})", P(&x0), P(&x1));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() | a1.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_bitxor(x0: K<QIRBigInt>, x1: K<QIRBigInt>) -> K<QIRBigInt> {
    trace!("calling qir_shim_rt_bigint_bitxor({}, {})", P(&x0), P(&x1));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() ^ a1.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_create_array(x0: i32, x1: *mut i8) -> K<QIRBigInt> {
    trace!("calling qir_shim_rt_bigint_create_array({}, {:?})", x0, x1);
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    ctx.add(QIRBigInt::from_byte_array(unsafe {
        core::slice::from_raw_parts(x1 as *const _, x0 as usize)
    }))
}
pub fn isq_qir_shim_rt_bigint_create_i64(x0: i64) -> K<QIRBigInt> {
    trace!("calling qir_shim_rt_bigint_create_i64({})", x0);
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    ctx.add(QIRBigInt::from_i64(x0))
}
pub fn isq_qir_shim_rt_bigint_divide(x0: K<QIRBigInt>, x1: K<QIRBigInt>) -> K<QIRBigInt> {
    trace!("calling qir_shim_rt_bigint_divide({}, {})", P(&x0), P(&x1));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() / a1.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_equal(x0: K<QIRBigInt>, x1: K<QIRBigInt>) -> bool {
    trace!("calling qir_shim_rt_bigint_equal({}, {})", P(&x0), P(&x1));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    a0.get_bigint() == a1.get_bigint()
}
pub fn isq_qir_shim_rt_bigint_get_data(x0: K<QIRBigInt>) -> *mut i8 {
    trace!("calling qir_shim_rt_bigint_get_data({})", P(&x0));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    a0.get_raw().as_ptr() as *mut _
}
pub fn isq_qir_shim_rt_bigint_get_length(x0: K<QIRBigInt>) -> i32 {
    trace!("calling qir_shim_rt_bigint_get_length({})", P(&x0));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    a0.get_raw().len() as i32
}
pub fn isq_qir_shim_rt_bigint_greater(x0: K<QIRBigInt>, x1: K<QIRBigInt>) -> bool {
    trace!("calling qir_shim_rt_bigint_greater({}, {})", P(&x0), P(&x1));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    a0.get_bigint() > a1.get_bigint()
}
pub fn isq_qir_shim_rt_bigint_greater_eq(x0: K<QIRBigInt>, x1: K<QIRBigInt>) -> bool {
    trace!(
        "calling qir_shim_rt_bigint_greater_eq({}, {})",
        P(&x0),
        P(&x1)
    );
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    a0.get_bigint() >= a1.get_bigint()
}
pub fn isq_qir_shim_rt_bigint_modulus(x0: K<QIRBigInt>, x1: K<QIRBigInt>) -> K<QIRBigInt> {
    trace!("calling qir_shim_rt_bigint_modulus({}, {})", P(&x0), P(&x1));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() % a1.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_multiply(x0: K<QIRBigInt>, x1: K<QIRBigInt>) -> K<QIRBigInt> {
    trace!(
        "calling qir_shim_rt_bigint_multiply({}, {})",
        P(&x0),
        P(&x1)
    );
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() * a1.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_negate(x0: K<QIRBigInt>) -> K<QIRBigInt> {
    trace!("calling qir_shim_rt_bigint_negate({})", P(&x0));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a = QIRBigInt::from_bigint(-a0.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_power(x0: K<QIRBigInt>, x1: i32) -> K<QIRBigInt> {
    trace!("calling qir_shim_rt_bigint_power({}, {})", P(&x0), x1);
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint().pow(x1 as u32));
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_shiftleft(x0: K<QIRBigInt>, x1: i64) -> K<QIRBigInt> {
    trace!("calling qir_shim_rt_bigint_shiftleft({}, {})", P(&x0), x1);
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() << x1);
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_shiftright(x0: K<QIRBigInt>, x1: i64) -> K<QIRBigInt> {
    trace!("calling qir_shim_rt_bigint_shiftright({}, {})", P(&x0), x1);
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() >> x1);
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_subtract(x0: K<QIRBigInt>, x1: K<QIRBigInt>) -> K<QIRBigInt> {
    trace!(
        "calling qir_shim_rt_bigint_subtract({}, {})",
        P(&x0),
        P(&x1)
    );
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() - a1.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_to_string(x0: K<QIRBigInt>) -> K<QIRString> {
    trace!("calling qir_shim_rt_bigint_to_string({})", P(&x0));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let s = QIRString::from_str(&format!("{}", a0.get_bigint()));
    ctx.add(s)
}
pub fn isq_qir_shim_rt_bigint_update_reference_count(x0: K<QIRBigInt>, x1: i32) -> () {
    trace!(
        "calling qir_shim_rt_bigint_update_reference_count({}, {})",
        P(&x0),
        x1
    );
    if x0.is_null() {
        return;
    }
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    x0.update_ref_count(&ctx, x1 as isize);
}
pub fn isq_qir_shim_rt_bool_to_string(x0: bool) -> K<QIRString> {
    trace!("calling qir_shim_rt_bool_to_string({})", x0);
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let s = QIRString::from_str(&format!("{}", x0));
    ctx.add(s)
}
pub fn isq_qir_shim_rt_callable_copy(x0: K<QIRCallable>, x1: bool) -> K<QIRCallable> {
    trace!("calling qir_shim_rt_callable_copy({}, {})", P(&x0), x1);
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    x0.try_copy(&ctx, x1)
}
// pub fn isq_qir_shim_rt_callable_invoke(
//     x0: K<QIRCallable>,
//     x1: TupleBodyPtr,
//     x2: TupleBodyPtr,
// ) -> () {
//     trace!(
//         "calling qir_shim_rt_callable_invoke({}, {}, {})",
//         P(&x0),
//         x1 as usize,
//         x2 as usize
//     );
//     let rctx = context();
//     let mut ctx = rctx.lock().unwrap();
//     let callable = x0.get(&ctx);
//     let callable_borrow = RefCell::borrow(&callable);
//     let f = callable_borrow.defer_invoke();
//     drop(callable_borrow);
//     drop(callable);
//     drop(ctx);
//     drop(rctx);
//     (f)(x1, x2);
//     trace!("returning from qir_shim_rt_callable_invoke");
// }
pub fn isq_qir_shim_rt_callable_make_adjoint(x0: K<QIRCallable>) -> () {
    trace!("calling qir_shim_rt_callable_make_adjoint({})", P(&x0));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    x0.get(&ctx).borrow_mut().make_adjoint();
}
pub fn isq_qir_shim_rt_callable_make_controlled(x0: K<QIRCallable>) -> () {
    trace!("calling qir_shim_rt_callable_make_controlled({})", P(&x0));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    x0.get(&ctx).borrow_mut().make_controlled();
}
pub fn isq_qir_shim_rt_callable_update_alias_count(x0: K<QIRCallable>, x1: i32) -> () {
    trace!(
        "calling qir_shim_rt_callable_update_alias_count({}, {})",
        P(&x0),
        x1
    );
    if x0.is_null() {
        return;
    }
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    x0.update_alias_count(&ctx, x1 as isize);
}
pub fn isq_qir_shim_rt_callable_update_reference_count(x0: K<QIRCallable>, x1: i32) -> () {
    trace!(
        "calling qir_shim_rt_callable_update_reference_count({}, {})",
        P(&x0),
        x1
    );
    if x0.is_null() {
        return;
    }
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    x0.update_ref_count(&ctx, x1 as isize);
}
pub fn isq_qir_shim_rt_capture_update_alias_count(x0: K<QIRCallable>, x1: i32) -> () {
    trace!(
        "calling qir_shim_rt_capture_update_alias_count({}, {})",
        P(&x0),
        x1
    );
    if x0.is_null() {
        return;
    }
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let f = x0.get(&ctx).borrow_mut().defer_capture_update_alias_count(x1);
    drop(ctx);
    (f)();
}
pub fn isq_qir_shim_rt_capture_update_reference_count(x0: K<QIRCallable>, x1: i32) -> () {
    trace!(
        "calling qir_shim_rt_capture_update_reference_count({}, {})",
        P(&x0),
        x1
    );
    if x0.is_null() {
        return;
    }
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let f = x0.get(&ctx).borrow_mut().defer_capture_update_ref_count(x1);
    drop(ctx);
    (f)();
}
pub fn isq_qir_shim_rt_double_to_string(x0: f64) -> K<QIRString> {
    trace!("calling qir_shim_rt_double_to_string({})", x0);
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let s = QIRString::from_str(&format!("{}", x0));
    ctx.add(s)
}
pub fn isq_qir_shim_rt_fail(x0: K<QIRString>) -> () {
    trace!("calling qir_shim_rt_fail({})", P(&x0));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let s = x0.get(&ctx);
    let r = s.get_str();
    panic!("Fail: {}", r);
}
pub fn isq_qir_shim_rt_int_to_string(x0: i64) -> K<QIRString> {
    trace!("calling qir_shim_rt_int_to_string({})", x0);
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let s = QIRString::from_str(&format!("{}", x0));
    ctx.add(s)
}
pub fn isq_qir_shim_rt_message(x0: K<QIRString>) -> () {
    trace!("calling qir_shim_rt_message({})", P(&x0));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let s = x0.get(&ctx);
    let r = s.get_str();
    ctx.message(r);
}
pub fn isq_qir_shim_rt_pauli_to_string(x0: QIRPauli) -> K<QIRString> {
    trace!("calling qir_shim_rt_pauli_to_string({:?})", x0);
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let s = QIRString::from_str(&format!("{:?}", x0));
    ctx.add(s)
}
pub fn isq_qir_shim_rt_qubit_allocate() -> K<QIRQubit> {
    trace!("calling qir_shim_rt_qubit_allocate()");
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let q = ctx.get_device_mut().alloc_qubit();
    unsafe { core::mem::transmute(q) }
}
pub fn isq_qir_shim_rt_qubit_allocate_array(x0: i32) -> K<QIRArray> {
    trace!("calling qir_shim_rt_qubit_allocate_array({})", x0);
    let array = isq_qir_shim_rt_array_create_1d(core::mem::size_of::<usize>() as i32, x0 as i64);
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let mut qubits = Vec::new();
    for i in 0..x0 {
        qubits.push(ctx.get_device_mut().alloc_qubit());
    }
    let mut arr = array.get_mut(&mut ctx);
    let data = arr.get_data_mut();
    for i in 0..x0 as usize {
        data[i] = qubits[i];
    }
    array
}
pub fn isq_qir_shim_rt_qubit_release(x0: K<QIRQubit>) -> () {
    trace!("calling qir_shim_rt_qubit_release({})", P(&x0));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    ctx.get_device_mut()
        .free_qubit(unsafe { core::mem::transmute(x0) });
}
pub fn isq_qir_shim_rt_qubit_release_array(x0: K<QIRArray>) -> () {
    trace!("calling qir_shim_rt_qubit_release_array({})", P(&x0));
    if x0.is_null() {
        panic!("Attempted to release null qubit array");
    }
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let arr = x0.get(&mut ctx);
    let data = arr.get_1d_data_of::<usize>();
    let mut qubits = Vec::new();
    for i in 0..data.len() {
        qubits.push(data[i]);
    }
    drop(arr);
    for qubit in qubits.iter().copied() {
        ctx.get_device_mut().free_qubit(qubit);
    }
    x0.update_ref_count(&ctx, -1)
}
pub fn isq_qir_shim_rt_qubit_to_string(x0: K<QIRQubit>) -> K<QIRString> {
    trace!("calling qir_shim_rt_qubit_to_string({})", P(&x0));
    let qubit_id: usize = unsafe { core::mem::transmute(x0) };
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let s = QIRString::from_str(&format!("qubit<{:?}>", qubit_id));
    ctx.add(s)
}
pub fn isq_qir_shim_rt_range_to_string(start: i64, step: i64, end: i64) -> K<QIRString> {
    let x0 = QIRRange{start, step, end};
    trace!("calling qir_shim_rt_range_to_string({:?})", &unsafe { x0 });
    let r = x0;
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let s = QIRString::from_str(&format!("{:?}", r));
    ctx.add(s)
}
pub fn isq_qir_shim_rt_result_equal(x0: QIRResult, x1: QIRResult) -> bool {
    trace!("calling qir_shim_rt_result_equal({:?}, {:?})", x0, x1);
    x0 == x1
}
pub fn isq_qir_shim_rt_result_get_one() -> QIRResult {
    trace!("calling qir_shim_rt_result_get_one()");
    QIR_RESULT_ONE
}
pub fn isq_qir_shim_rt_result_get_zero() -> QIRResult {
    trace!("calling qir_shim_rt_result_get_zero()");
    QIR_RESULT_ZERO
}
pub fn isq_qir_shim_rt_result_to_string(x0: QIRResult) -> K<QIRString> {
    trace!("calling qir_shim_rt_result_to_string({:?})", x0);
    let r = if x0 == QIR_RESULT_ONE { "One" } else { "Zero" };
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let s = QIRString::from_str(&format!("{:?}", r));
    ctx.add(s)
}
pub fn isq_qir_shim_rt_result_update_reference_count(x0: QIRResult, x1: i32) -> () {
    trace!(
        "calling qir_shim_rt_result_update_reference_count({:?}, {})",
        x0,
        x1
    );
    // Do nothing.
}
pub fn isq_qir_shim_rt_string_concatenate(x0: K<QIRString>, x1: K<QIRString>) -> K<QIRString> {
    trace!(
        "calling qir_shim_rt_string_concatenate({}, {})",
        P(&x0),
        P(&x1)
    );
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRString::from_str(&format!("{}{}", a0.get_str(), a1.get_str()));
    drop(a0);
    drop(a1);
    ctx.add(a)
}
pub fn isq_qir_shim_rt_string_create(x0: *mut i8) -> K<QIRString> {
    trace!("calling qir_shim_rt_string_create({:?})", x0);
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a = unsafe { QIRString::from_i8_array(x0 as *const i8) };
    ctx.add(a)
}
pub fn isq_qir_shim_rt_string_equal(x0: K<QIRString>, x1: K<QIRString>) -> bool {
    trace!("calling qir_shim_rt_string_equal({}, {})", P(&x0), P(&x1));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    a0.get_str() == a1.get_str()
}
pub fn isq_qir_shim_rt_string_get_data(x0: K<QIRString>) -> *mut i8 {
    trace!("calling qir_shim_rt_string_get_data({})", P(&x0));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    a0.get_raw().as_ptr() as *mut i8
}
pub fn isq_qir_shim_rt_string_get_length(x0: K<QIRString>) -> i32 {
    trace!("calling qir_shim_rt_string_get_length({})", P(&x0));
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let a0 = x0.get(&ctx);
    a0.get_raw().len() as i32
}
pub fn isq_qir_shim_rt_string_update_reference_count(x0: K<QIRString>, x1: i32) -> () {
    trace!(
        "calling qir_shim_rt_string_update_reference_count({}, {})",
        P(&x0),
        x1
    );
    if x0.is_null() {
        return;
    }
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    x0.update_ref_count(&ctx, x1 as isize)
}
pub fn isq_qir_shim_rt_tuple_copy(x0: TupleBodyPtr, x1: bool) -> TupleBodyPtr {
    trace!("calling qir_shim_rt_tuple_copy({:?}, {})", x0, x1);
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let tup = QIRTuple::from_body(x0);
    let new_tuple = tup.try_copy(&ctx, x1);
    let t = new_tuple.get(&ctx);
    t.to_body()
}
pub fn isq_qir_shim_rt_tuple_create(x0: i64) -> TupleBodyPtr {
    trace!("calling qir_shim_rt_tuple_create({})", x0);
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let new_tuple = QIRTuple::new(x0 as usize);
    let id = ctx.add(new_tuple);
    let current_tuple: Ref<QIRTuple> = id.get(&ctx);
    current_tuple.set_resource_id(id.key);
    current_tuple.to_body()
}
pub fn isq_qir_shim_rt_tuple_update_alias_count(x0: TupleBodyPtr, x1: i32) -> () {
    trace!(
        "calling qir_shim_rt_tuple_update_alias_count({:?}, {})",
        x0,
        x1
    );
    if x0.is_null() {
        return;
    }
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let tup_key = QIRTuple::from_body(x0);
    tup_key.update_alias_count(&ctx, x1 as isize);
}
pub fn isq_qir_shim_rt_tuple_update_reference_count(x0: TupleBodyPtr, x1: i32) -> () {
    trace!(
        "calling qir_shim_rt_tuple_update_reference_count({:?}, {})",
        x0,
        x1
    );
    if x0.is_null() {
        return;
    }
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let tup_key = QIRTuple::from_body(x0);
    tup_key.update_ref_count(&ctx, x1 as isize);
}
