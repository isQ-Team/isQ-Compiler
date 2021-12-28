use core::cell::RefCell;

use alloc::{boxed::Box, rc::Rc};

use crate::{devices::checked::{NumberedQDevice}};

use super::resource::ResourceMap;

pub struct QIRContext{
    device: Box<dyn NumberedQDevice>,
    classical_resource_manager: ResourceMap,
}

impl QIRContext{
    pub fn new(device: Box<dyn NumberedQDevice>) -> Self{
        Self{
            device,
            classical_resource_manager: ResourceMap::new(),
        }
    }
    pub fn get_device(&self) -> &dyn NumberedQDevice{
        &*self.device
    }
    pub fn get_device_mut(&mut self) -> &mut dyn NumberedQDevice{
        &mut *self.device
    }
    pub fn get_classical_resource_manager(&self) -> &ResourceMap{
        &self.classical_resource_manager
    }
    pub fn get_classical_resource_manager_mut(&mut self) -> &mut ResourceMap{
        &mut self.classical_resource_manager
    }
}


#[thread_local]
static mut QIR_CURRENT_CONTEXT: Option<Rc<RefCell<QIRContext>>> = None;

pub fn make_context_current(a: Rc<RefCell<QIRContext>>){
    unsafe{
        QIR_CURRENT_CONTEXT = Some(a);
    }
}

pub fn get_current_context() -> Rc<RefCell<QIRContext>>{
    unsafe{
        QIR_CURRENT_CONTEXT.as_ref().expect("No QIR context made current").clone()
    }
}