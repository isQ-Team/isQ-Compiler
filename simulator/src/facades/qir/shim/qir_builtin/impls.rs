use core::borrow::Borrow;
use core::cell::{RefCell, Ref, RefMut};

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::facades::qir::bigint::QBigInt;
use crate::facades::qir::callable::QCallable;
use crate::facades::qir::resource::{AliasingTracker, ResourceManager};

use super::types::*;
use super::super::super::context::get_current_context as context;
pub fn isq_qir_shim_rt_array_concatenate(x0: K<QIRArray>, x1: K<QIRArray>)->K<QIRArray> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = a0.concat_1d(a1.borrow());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_array_copy(x0: K<QIRArray>, x1: bool)->K<QIRArray> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    x0.try_copy(&ctx, x1)
}
pub fn isq_qir_shim_rt_array_create(size: i32, num_dims: i32, pointer: *mut i64)->K<QIRArray> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let dimensions = unsafe {core::slice::from_raw_parts_mut(pointer as *mut usize, num_dims as usize)};
    let a = QIRArray::new(size as usize, dimensions);
    ctx.add(a)
}
pub fn isq_qir_shim_rt_array_create_1d(x0: i32, mut x1: i64)->K<QIRArray> {
    isq_qir_shim_rt_array_create(x0, 1, &mut x1 as *mut _)
}
pub fn isq_qir_shim_rt_array_get_dim(x0: K<QIRArray>)->i32 {
    let rctx = context();
    let ctx = RefCell::borrow(&*rctx);
    let x = x0.get(&ctx).get_dimensions().len() as i32;
    x
}
pub fn isq_qir_shim_rt_array_get_element_ptr(x0: K<QIRArray>, x1: *mut i64)->*mut i8 {
    let rctx = context();
    let ctx = RefCell::borrow(&*rctx);
    let indices = unsafe {core::slice::from_raw_parts_mut(x1, x0.get(&ctx).get_dimensions().len())};
    let x= x0.get(&ctx).get_element(indices);
    x
}
pub fn isq_qir_shim_rt_array_get_element_ptr_1d(x0: K<QIRArray>, x1: i64)->*mut i8 {
    let mut x2 = x1;
    isq_qir_shim_rt_array_get_element_ptr(x0, &mut x2 as *mut _)
}
pub fn isq_qir_shim_rt_array_get_size(x0: K<QIRArray>, x1: i32)->i64 {
    let rctx = context();
    let ctx = RefCell::borrow(&*rctx);
    let x = x0.get(&ctx).get_dimensions()[x1 as usize] as i64;
    x
}
pub fn isq_qir_shim_rt_array_get_size_1d(x0: K<QIRArray>)->i64 {
    let rctx = context();
    let ctx = rctx.borrow_mut();
    let a0 = x0.get(&ctx);
    if a0.get_dimensions().len()!=1{
        panic!("Array has {} dimensions, expected 1", a0.get_dimensions().len());
    }
    a0.get_dimensions()[0] as i64
}
pub fn isq_qir_shim_rt_array_project(x0: K<QIRArray>, index: i32, id: i64, x3: bool)->K<QIRArray> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a = a0.project(index as usize,id as usize);
    // special handling for array.
    if a0.get_alias_count()==0 && !x3{
        drop(a0);
        drop(ctx);
        let ctx = RefCell::borrow_mut(&rctx);
        let mut a0 = x0.get_mut(&ctx);
        *a0 = a;
        x0.update_ref_count(&ctx, 1);
        x0
    }else{
        ctx.add(a)
    }
}
pub fn isq_qir_shim_rt_array_slice(x0: K<QIRArray>, x1: i32, x2: *const QIRRange, x3: bool)->K<QIRArray> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a = a0.slice(x1 as usize,unsafe {*x2});
    // special handling for array.
    if a0.get_alias_count()==0 && !x3{
        drop(a0);
        drop(ctx);
        let ctx = RefCell::borrow_mut(&rctx);
        let mut a0 = x0.get_mut(&ctx);
        *a0 = a;
        x0.update_ref_count(&ctx, 1);
        x0
    }else{
        ctx.add(a)
    }
}
pub fn isq_qir_shim_rt_array_slice_1d(x0: K<QIRArray>, x1: *const QIRRange, x2: bool)->K<QIRArray> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    if a0.get_dimensions().len()!=1{
        panic!("Array has {} dimensions, expected 1", a0.get_dimensions().len());
    }
    drop(a0);
    drop(ctx);
    drop(rctx);
    isq_qir_shim_rt_array_slice(x0, 0, x1, x2)
}
pub fn isq_qir_shim_rt_array_update_alias_count(x0: K<QIRArray>, x1: i32)->() {
    if x0.is_null(){
        return
    }
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    x0.update_alias_count(&ctx, x1 as isize)
}
pub fn isq_qir_shim_rt_array_update_reference_count(x0: K<QIRArray>, x1: i32)->() {
    if x0.is_null(){
        return
    }
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    x0.update_ref_count(&ctx, x1 as isize)
}
pub fn isq_qir_shim_rt_bigint_add(x0: K<QIRBigInt>, x1: K<QIRBigInt>)->K<QIRBigInt> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() + a1.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_bitand(x0: K<QIRBigInt>, x1: K<QIRBigInt>)->K<QIRBigInt> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() & a1.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_bitor(x0: K<QIRBigInt>, x1: K<QIRBigInt>)->K<QIRBigInt> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() | a1.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_bitxor(x0: K<QIRBigInt>, x1: K<QIRBigInt>)->K<QIRBigInt> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() ^ a1.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_create_array(x0: i32, x1: *mut i8)->K<QIRBigInt> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    ctx.add(QIRBigInt::from_byte_array(unsafe{
        core::slice::from_raw_parts(x1 as *const _, x0 as usize)
    }))
}
pub fn isq_qir_shim_rt_bigint_create_i64(x0: i64)->K<QIRBigInt> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    ctx.add(QIRBigInt::from_i64(x0))
}
pub fn isq_qir_shim_rt_bigint_divide(x0: K<QIRBigInt>, x1: K<QIRBigInt>)->K<QIRBigInt> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() / a1.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_equal(x0: K<QIRBigInt>, x1: K<QIRBigInt>)->bool {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    a0.get_bigint()==a1.get_bigint()
}
pub fn isq_qir_shim_rt_bigint_get_data(x0: K<QIRBigInt>)->*mut i8 {
    let rctx = context();
    let ctx = rctx.borrow_mut();
    let a0 = x0.get(&ctx);
    a0.get_raw().as_ptr() as *mut _
}
pub fn isq_qir_shim_rt_bigint_get_length(x0: K<QIRBigInt>)->i32 {
    let rctx = context();
    let ctx = rctx.borrow_mut();
    let a0 = x0.get(&ctx);
    a0.get_raw().len() as i32
}
pub fn isq_qir_shim_rt_bigint_greater(x0: K<QIRBigInt>, x1: K<QIRBigInt>)->bool {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    a0.get_bigint()>a1.get_bigint()
}
pub fn isq_qir_shim_rt_bigint_greater_eq(x0: K<QIRBigInt>, x1: K<QIRBigInt>)->bool {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    a0.get_bigint()>=a1.get_bigint()
}
pub fn isq_qir_shim_rt_bigint_modulus(x0: K<QIRBigInt>, x1: K<QIRBigInt>)->K<QIRBigInt> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() % a1.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_multiply(x0: K<QIRBigInt>, x1: K<QIRBigInt>)->K<QIRBigInt> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() * a1.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_negate(x0: K<QIRBigInt>)->K<QIRBigInt> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a = QIRBigInt::from_bigint(-a0.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_power(x0: K<QIRBigInt>, x1: i32)->K<QIRBigInt> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint().pow(x1 as u32));
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_shiftleft(x0: K<QIRBigInt>, x1: i64)->K<QIRBigInt> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint()<<x1);
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_shiftright(x0: K<QIRBigInt>, x1: i64)->K<QIRBigInt> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint()>>x1);
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_subtract(x0: K<QIRBigInt>, x1: K<QIRBigInt>)->K<QIRBigInt> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRBigInt::from_bigint(a0.get_bigint() - a1.get_bigint());
    ctx.add(a)
}
pub fn isq_qir_shim_rt_bigint_to_string(x0: K<QIRBigInt>)->K<QIRString> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let s = QIRString::from_str(&format!("{}", a0.get_bigint()));
    ctx.add(s)
}
pub fn isq_qir_shim_rt_bigint_update_reference_count(x0: K<QIRBigInt>, x1: i32)->() {
    if x0.is_null(){
        return
    }
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    x0.update_ref_count(&ctx, x1 as isize);
}
pub fn isq_qir_shim_rt_bool_to_string(x0: bool)->K<QIRString> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let s = QIRString::from_str(&format!("{}", x0));
    ctx.add(s)
}
pub fn isq_qir_shim_rt_callable_copy(x0: K<QIRCallable>, x1: bool)->K<QIRCallable> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    x0.try_copy(&ctx, x1)
}
pub fn isq_qir_shim_rt_callable_create(x0: *mut i8, x1: *mut i8, x2: *mut i8, x3: *mut i8, x4: *mut i8, x5: *mut i8, x6: TupleBodyPtr)->K<QIRCallable> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    ctx.add(RefCell::new(QCallable::new(
        &[x0 as _, x1 as _, x2 as _, x3 as _],
        Some(&[x4 as _, x5 as _]),
        x6
    )))
}
pub fn isq_qir_shim_rt_callable_invoke(x0: K<QIRCallable>, x1: TupleBodyPtr, x2: TupleBodyPtr)->() {
    let rctx = context();
    let mut ctx = RefCell::borrow(&*rctx);
    let callable = x0.get(&ctx);
    let callable_borrow = RefCell::borrow(&callable);
    callable_borrow.invoke(x1, x2);
}
pub fn isq_qir_shim_rt_callable_make_adjoint(x0: K<QIRCallable>)->() {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    x0.get(&ctx).borrow_mut().make_adjoint();
}
pub fn isq_qir_shim_rt_callable_make_controlled(x0: K<QIRCallable>)->() {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    x0.get(&ctx).borrow_mut().make_controlled();
}
pub fn isq_qir_shim_rt_callable_update_alias_count(x0: K<QIRCallable>, x1: i32)->() {
    if x0.is_null(){
        return
    }
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    x0.update_alias_count(&ctx, x1 as isize);
}
pub fn isq_qir_shim_rt_callable_update_reference_count(x0: K<QIRCallable>, x1: i32)->() {
    if x0.is_null(){
        return
    }
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    x0.update_ref_count(&ctx, x1 as isize);
}
pub fn isq_qir_shim_rt_capture_update_alias_count(x0: K<QIRCallable>, x1: i32)->() {
    if x0.is_null(){
        return
    }
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    x0.get(&ctx).borrow_mut().capture_update_alias_count(x1);
}
pub fn isq_qir_shim_rt_capture_update_reference_count(x0: K<QIRCallable>, x1: i32)->() {
    if x0.is_null(){
        return
    }
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    x0.get(&ctx).borrow_mut().capture_update_ref_count(x1);
}
pub fn isq_qir_shim_rt_double_to_string(x0: f64)->K<QIRString> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let s = QIRString::from_str(&format!("{}", x0));
    ctx.add(s)
}
pub fn isq_qir_shim_rt_fail(x0: K<QIRString>)->() {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let s = x0.get(&ctx);
    let r = s.get_str();
    panic!("Fail: {}", r);
}
pub fn isq_qir_shim_rt_int_to_string(x0: i64)->K<QIRString> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let s = QIRString::from_str(&format!("{}", x0));
    ctx.add(s)
}
pub fn isq_qir_shim_rt_message(x0: K<QIRString>)->(){
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let s = x0.get(&ctx);
    let r = s.get_str();
    ctx.message(r);
}
pub fn isq_qir_shim_rt_pauli_to_string(x0: QIRPauli)->K<QIRString> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let s = QIRString::from_str(&format!("{:?}", x0));
    ctx.add(s)
}
pub fn isq_qir_shim_rt_qubit_allocate()->K<QIRQubit> {
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    let q = ctx.get_device_mut().alloc_qubit();
    unsafe {core::mem::transmute(q)}
}
pub fn isq_qir_shim_rt_qubit_allocate_array(x0: i32)->K<QIRArray> {
    let array = isq_qir_shim_rt_array_create_1d(core::mem::size_of::<usize>() as i32, x0 as i64);
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    let mut qubits = Vec::new();
    for i in 0..x0{
        qubits.push(ctx.get_device_mut().alloc_qubit());
    }
    let mut arr = array.get_mut(&mut ctx);
    let data = arr.get_data_mut();
    for i in 0..x0 as usize{
        data[i] = qubits[i];
    }
    array
}
pub fn isq_qir_shim_rt_qubit_release(x0: K<QIRQubit>)->() {
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    ctx.get_device_mut().free_qubit(unsafe {core::mem::transmute(x0)});
}
pub fn isq_qir_shim_rt_qubit_release_array(x0: K<QIRArray>)->() {
    if x0.is_null(){
        panic!("Attempted to release null qubit array");
    }
    let rctx = context();
    let mut ctx = RefCell::borrow_mut(&rctx);
    let arr = x0.get(&mut ctx);
    let data = arr.get_1d_data_of::<usize>();
    let mut qubits = Vec::new();
    for i in 0..data.len(){
        qubits.push(data[i]);
    }
    drop(arr);
    for qubit in qubits.iter().copied(){
        ctx.get_device_mut().free_qubit(qubit);
    }
    x0.update_ref_count(&ctx, -1)
}
pub fn isq_qir_shim_rt_qubit_to_string(x0: K<QIRQubit>)->K<QIRString> {
    let qubit_id: usize = unsafe {core::mem::transmute(x0)};
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let s = QIRString::from_str(&format!("qubit<{:?}>", qubit_id));
    ctx.add(s)
}
pub fn isq_qir_shim_rt_range_to_string(x0: *const QIRRange)->K<QIRString> {
    let r = unsafe {*x0};
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let s = QIRString::from_str(&format!("{:?}", r));
    ctx.add(s)
}
pub fn isq_qir_shim_rt_result_equal(x0: QIRResult, x1: QIRResult)->bool {
    x0==x1
}
pub fn isq_qir_shim_rt_result_get_one()->QIRResult {
    QIR_RESULT_ONE
}
pub fn isq_qir_shim_rt_result_get_zero()->QIRResult {
    QIR_RESULT_ZERO
}
pub fn isq_qir_shim_rt_result_to_string(x0: QIRResult)->K<QIRString> {
    let r = if x0==QIR_RESULT_ONE {"One"} else {"Zero"};
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let s = QIRString::from_str(&format!("{:?}", r));
    ctx.add(s)
}
pub fn isq_qir_shim_rt_result_update_reference_count(_x0: QIRResult, _x1: i32)->() {
    // Do nothing.
}
pub fn isq_qir_shim_rt_string_concatenate(x0: K<QIRString>, x1: K<QIRString>)->K<QIRString> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    let a = QIRString::from_str(&format!("{}{}", a0.get_str(), a1.get_str()));
    ctx.add(a)
}
pub fn isq_qir_shim_rt_string_create(x0: *mut i8)->K<QIRString> {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a = unsafe {QIRString::from_i8_array(x0 as *const i8)};
    ctx.add(a)
}
pub fn isq_qir_shim_rt_string_equal(x0: K<QIRString>, x1: K<QIRString>)->bool {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    let a1 = x1.get(&ctx);
    a0.get_str() == a1.get_str()
}
pub fn isq_qir_shim_rt_string_get_data(x0: K<QIRString>)->*mut i8 {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    a0.get_raw().as_ptr() as *mut i8
}
pub fn isq_qir_shim_rt_string_get_length(x0: K<QIRString>)->i32 {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let a0 = x0.get(&ctx);
    a0.get_raw().len() as i32
}
pub fn isq_qir_shim_rt_string_update_reference_count(x0: K<QIRString>, x1: i32)->() {
    if x0.is_null(){
        return
    }
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    x0.update_ref_count(&ctx, x1 as isize)
}
pub fn isq_qir_shim_rt_tuple_copy(x0: TupleBodyPtr, x1: bool)->TupleBodyPtr {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let tup = QIRTuple::from_body(x0); 
    let new_tuple= tup.try_copy(&ctx, x1);
    let t= new_tuple.get(&ctx);
    t.to_body()
}
pub fn isq_qir_shim_rt_tuple_create(x0: i64)->TupleBodyPtr {
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let new_tuple = QIRTuple::new(x0 as usize);
    let id = ctx.add(new_tuple);
    let current_tuple: Ref<QIRTuple> = id.get(&ctx);
    current_tuple.set_resource_id(id.key);
    current_tuple.to_body()

}
pub fn isq_qir_shim_rt_tuple_update_alias_count(x0: TupleBodyPtr, x1: i32)->() {
    if x0.is_null(){
        return
    }
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let tup_key = QIRTuple::from_body(x0);
    tup_key.update_alias_count(&ctx, x1 as isize);
}
pub fn isq_qir_shim_rt_tuple_update_reference_count(x0: TupleBodyPtr, x1: i32)->() {
    if x0.is_null(){
        return
    }
    let rctx = context();
    let ctx = RefCell::borrow(&rctx);
    let tup_key = QIRTuple::from_body(x0);
    tup_key.update_ref_count(&ctx, x1 as isize);
}
