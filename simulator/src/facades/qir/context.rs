extern crate std;
use std::collections::HashSet;
use core::any::Any;
use std::sync::Mutex;

use alloc::sync::Arc;
use alloc::boxed::Box;

use crate::qdevice::QDevice;

use super::resource::ResourceManagerExt;
use super::resource::ResourceMap;
use super::shim::qsharp_foundation::types::QIRArray;
pub struct QIRContext {
    device: Box<dyn QDevice<Qubit = usize>>,
    classical_resource_manager: ResourceMap,
    message_handler: Box<dyn Fn(&str) -> ()>,
    disabled_bp : HashSet<i64>,
}
use crate::facades::qir::resource::ResourceKey;
impl QIRContext {
    pub fn new(
        device: Box<dyn QDevice<Qubit = usize>>,
        message_handler: Box<dyn Fn(&str) -> ()>,
    ) -> Self {
        Self {
            device,
            classical_resource_manager: ResourceMap::new(),
            message_handler,
            disabled_bp: HashSet::new(),
        }
    }
    pub fn get_device(&self) -> &dyn QDevice<Qubit = usize> {
        &*self.device
    }
    pub fn get_device_mut(&mut self) -> &mut dyn QDevice<Qubit = usize> {
        &mut *self.device
    }
    pub fn get_classical_resource_manager(&self) -> &ResourceMap {
        &self.classical_resource_manager
    }
    pub fn get_classical_resource_manager_mut(&mut self) -> &mut ResourceMap {
        &mut self.classical_resource_manager
    }
    pub fn add<T: Any + 'static>(&self, resource: T) -> ResourceKey<T> {
        self.classical_resource_manager.add_key(resource)
    }
    pub fn dump_machine(&mut self) {}
    pub fn dump_registers(&mut self, reg: ResourceKey<QIRArray>) {}
    pub fn message(&self, s: &str) {
        (self.message_handler)(s);
    }
    pub fn contains_bp(&self, index: i64) -> bool {
        self.disabled_bp.contains(&index)
    }
    pub fn disable_bp_index(&mut self, index: i64) {
        self.disabled_bp.insert(index);
    }
}

//#[thread_local]
static mut QIR_CURRENT_CONTEXT: Option<Arc<Mutex<QIRContext>>> = None;

pub fn make_context_current(a: Arc<Mutex<QIRContext>>) {
    unsafe {
        QIR_CURRENT_CONTEXT = Some(a);
    }
}

pub fn get_current_context() -> Arc<Mutex<QIRContext>> {
    unsafe {
        QIR_CURRENT_CONTEXT
            .as_ref()
            .expect("No QIR context made current")
            .clone()
    }
}
