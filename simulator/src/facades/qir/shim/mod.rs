pub mod shim_prelude{
    pub mod types{
        pub use super::super::super::types::*;
        pub use super::super::super::resource::ResourceKey as K;
        pub type QIRResult = QIRResultPtr;
        pub type TupleBodyPtr = QTupleContent;
        #[repr(C)]
        pub struct MeasurementProbabilityArgs{
            x0: K<QIRArray>,
            x1: K<QIRArray>,
            x2: QIRResult,
            x3: f64,
            x4: K<QIRString>,
            x5: f64
        }
    }
}

pub mod qir_builtin;
pub mod qsharp_foundation;
pub mod qsharp_core;