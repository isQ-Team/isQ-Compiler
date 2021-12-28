use core::cell::RefCell;

use super::{tuple::{QTupleContent}, resource::{AliasingTracker}};

pub type WrapperFn = extern "C" fn(capture: QTupleContent, args: QTupleContent, result: QTupleContent)->();
pub type MemManFn = extern "C" fn(tuple: QTupleContent, arg: i32)->();
const WRAPPER_NORMAL :usize = 0;
const WRAPPER_ADJOINT: usize = 1;
const WRAPPER_CONTROLLED: usize = 2;
const WRAPPER_CONTROLLED_ADJOINT: usize = 3;
#[derive(Clone)]
pub struct QCallable{
    function_table: [Option<WrapperFn>; 4],
    memory_management_table: [Option<MemManFn>; 2],
    capture: QTupleContent,
    alias_count: usize,
    is_adjoint: bool,
    is_controlled: bool,
}




impl AliasingTracker for QCallable{
    fn get_alias_count(&self)->usize{
        self.alias_count
    }

    fn full_copy(&self, _allocated_id: usize)->Self {
        let mut x = self.clone();
        x.alias_count=0;
        x
    }
}

impl QCallable{
    fn get_function_pointer_3<A1,A2,A3, O>(ptr: usize)->Option<extern "C" fn(A1,A2,A3)->O>{
        if ptr==0{
            return None;
        }
        unsafe {Some(core::mem::transmute(ptr))}
    }
    fn get_function_pointer_2<A1,A2, O>(ptr: usize)->Option<extern "C" fn(A1,A2)->O>{
        if ptr==0{
            return None;
        }
        unsafe {Some(core::mem::transmute(ptr))}
    }
    pub fn new(function_table: &[usize; 4], memory_management_table: Option<&[usize; 2]>, capture: QTupleContent)->Self{
        let mut transmuted_memory_management_table: [Option<MemManFn>; 2] = [None; 2];
        let mut transmuted_function_table: [Option<WrapperFn>; 4] = [None; 4];
        for i in 0..4{
            transmuted_function_table[i] = Self::get_function_pointer_3(function_table[i]);
        }
        if let Some(mmt) = memory_management_table{
            for i in 0..2{
                transmuted_memory_management_table[i] = Self::get_function_pointer_2(mmt[i]);
            }
        }
        QCallable{
            function_table: transmuted_function_table,
            memory_management_table: transmuted_memory_management_table,
            capture: capture,
            alias_count: 0,
            is_adjoint: false,
            is_controlled: false,
        }
    }
    fn get_function_id(&self)->usize{
        if self.is_adjoint{
            if self.is_controlled{
                WRAPPER_CONTROLLED_ADJOINT
            }else{
                WRAPPER_ADJOINT
            }
        }else{
            if self.is_controlled{
                WRAPPER_CONTROLLED
            }else{
                WRAPPER_NORMAL
            }
        }
    }
    fn get_function(&self)->WrapperFn{
        let f = self.function_table[self.get_function_id()];
        if let Some(x) = f{
            x
        } else {
            panic!("Wrapper function (adjoint = {}, controlled = {}) not defined.", self.is_adjoint, self.is_controlled);
        }
        
    }
    pub fn make_adjoint(&mut self){
        self.is_adjoint=true;
        self.get_function();
    }
    pub fn make_controlled(&mut self){
        self.is_controlled=true;
        self.get_function();
    }
    pub fn capture_update_ref_count(&self, val: i32){
        assert_ne!(self.capture, core::ptr::null_mut());
        if let Some(f) = self.memory_management_table[0]{
            f(self.capture, val);
        }
    }
    pub fn capture_update_alias_count(&self, val: i32){
        assert_ne!(self.capture, core::ptr::null_mut());
        if let Some(f) = self.memory_management_table[1]{
            f(self.capture, val);
        }
    }
    pub fn invoke(&self, args: QTupleContent, results: QTupleContent){
        self.get_function()(self.capture, args, results);
    }
}

pub type QIRCallable = RefCell<QCallable>;