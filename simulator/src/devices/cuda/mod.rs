mod qsim_kernel;
use qsim_kernel::qstate;

use self::qsim_kernel::{qstate_struct_size, qint_t, qstate_init, qstate_deinit};
pub struct QSimKernel{
    state: *mut qstate
}

impl QSimKernel{
    pub fn new(capacity: usize)->Self{
        unsafe{
            let mem = alloc::alloc::alloc(alloc::alloc::Layout::from_size_align(qstate_struct_size() as usize, qstate_struct_size() as usize).unwrap()) as *mut qstate;
            qstate_init(mem, capacity as qint_t);
            QSimKernel { state: mem }
        }
    }
}
impl Drop for QSimKernel{
    fn drop(&mut self){
        unsafe{
            qstate_deinit(self.state);
        }
        
    }
}