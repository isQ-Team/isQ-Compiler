use core::{alloc::Layout};

use alloc::{alloc::{alloc_zeroed, dealloc}};

use super::resource::AliasingTracker;

// Support for tuples.
// QIR spec does not make clear about the alignment of elements. We use machine-word alignment for now, i.e. same alignment as isize.
// Also QIR spec requires the tuple is represented in raw pointer instead of opaque pointers (i.e. no functions like tuple_get_data), preventing us from doing sanitizations.
// Bring your own valgrind.

// Dirty memory hacking, since we don't want to fight DST and its fatty pointer.

#[repr(C)]
pub struct QTupleOwned (*mut usize);

pub type QTupleContent = *mut u8;

const QTUPLE_SIZE_IN_USIZE: isize = 0;
const QTUPLE_SIZE_IN_BYTES: isize = 1;
const QTUPLE_RESMAN_ID: isize = 2;
const QTUPLE_ALIAS_COUNT: isize = 3;
const QTUPLE_DATA_START: isize = 4;
impl QTupleOwned{
    fn get_layout(size_in_usize: usize)->Layout{
        Layout::array::<usize>(size_in_usize+4).unwrap()
    }
    fn compute_size_in_usize(size_in_bytes: usize)->usize{
        (size_in_bytes + core::mem::size_of::<usize>()- 1)/core::mem::size_of::<usize>()
    }
    pub fn new(size_in_bytes: usize)->Self{
        let mem = unsafe {alloc_zeroed(Self::get_layout(Self::compute_size_in_usize(size_in_bytes)))};
        let s= Self(mem as *mut usize);
        s
    }
    pub fn to_body(&self)->QTupleContent{
        unsafe{(self.0).offset(QTUPLE_DATA_START) as _}
    }
    pub fn from_body(body: QTupleContent)->Self{
        let mem = unsafe{(body as *mut usize).offset(-QTUPLE_DATA_START)};
        Self(mem)
    }
    fn offset(&self, i: isize)->&usize{
        unsafe{&*self.0.offset(i)}
    }
    fn offset_mut(&self, i: isize)->&mut usize{
        unsafe{&mut *self.0.offset(i)}
    }
    pub fn size_in_usize(&self)->usize{
        *self.offset(QTUPLE_SIZE_IN_USIZE)
    }
    pub fn size_in_bytes(&self)->usize{
        *self.offset(QTUPLE_SIZE_IN_BYTES)
    }
    pub fn resource_id(&self)->usize{
        *self.offset(QTUPLE_RESMAN_ID)
    }
    pub fn set_resource_id(&mut self, id: usize){
        *self.offset_mut(QTUPLE_RESMAN_ID) = id;
    }
    pub fn update_alias_count(&self, delta: isize){
        let alias_count = self.offset_mut(QTUPLE_ALIAS_COUNT);
        let r = *alias_count as isize;
        if r+delta<0{
            panic!("QTupleRef: alias count underflow");
        }
        *alias_count = (r+delta) as usize;
    }
    pub fn get_alias_count_internal(&self)->usize{
        *self.offset(QTUPLE_ALIAS_COUNT)
    }
    unsafe fn free(&self) {
        let sz = self.size_in_usize();
        let layout = Self::get_layout(sz);
        dealloc(self.0 as *mut u8, layout);
    }
}

impl Drop for QIRTuple{
    fn drop(&mut self){
        unsafe{self.free()};
    }
}

impl AliasingTracker for QTupleOwned{
    fn get_alias_count(&self)->usize{
        self.get_alias_count_internal()
    }

    fn full_copy(&self, allocated_id: usize /* backdoor for tuples */)->Self {
        let sz = self.size_in_usize();
        let mut s = Self::new(sz);
        s.set_resource_id(allocated_id);
        s.update_alias_count(1);
        unsafe{
            core::ptr::copy_nonoverlapping(self.to_body(), s.to_body(), sz);
        }
        s
    }

}

pub type QIRTuple = QTupleOwned;