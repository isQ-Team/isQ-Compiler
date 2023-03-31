pub mod shim_prelude {
    pub mod types {
        use core::fmt::Display;

        pub use super::super::super::resource::ResourceKey as K;
        pub use super::super::super::types::*;
        pub type QIRResult = QIRResultPtr;
        pub type TupleBodyPtr = QTupleContent;
        #[repr(C)]
        pub struct MeasurementProbabilityArgs {
            x0: K<QIRArray>,
            x1: K<QIRArray>,
            x2: QIRResult,
            x3: f64,
            x4: K<QIRString>,
            x5: f64,
        }

        pub struct P<'a, T>(pub &'a T);
        impl<'a, U> Display for P<'a, K<U>> {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(
                    f,
                    "{}{{managed {}}}",
                    core::any::type_name::<U>(),
                    self.0.key
                )
            }
        }
    }
}
use core::cell::RefCell;

use super::context::get_current_context as context;
use crate::qdevice::QuantumOp;
use alloc::vec::Vec;
use itertools::Itertools;
use shim_prelude::types::*;
pub(crate) fn sq_op(controls: Option<K<QIRArray>>, op: QuantumOp, arg: &[f64], qubit: usize) {
    let rctx = context();
    let mut ctx = rctx.lock().unwrap();
    let controls: Vec<usize> = if let Some(x0) = controls {
        x0.get(&ctx)
            .get_1d_data_of::<usize>()
            .iter()
            .copied()
            .collect_vec()
    } else {
        vec![]
    };
    let controls_refs = controls.iter().collect_vec();
    let device = ctx.get_device_mut();
    device.controlled_qop(op, &controls_refs, &[&qubit], arg);
}

pub mod qir_builtin;
pub mod qsharp_core;
pub mod qsharp_foundation;
pub mod isq;