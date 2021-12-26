use alloc::boxed::Box;

use crate::qdevice::QDevice;

// Facade for QIR.
// We introduce the concept of "QIR Context" to localize QIR calls.
// A LLVM shim is provided so that a QIR program can be linked against the shim to form a shared library.
//                                                   +----------------------------+
//                                                   |                            |
// +---------------+       +----------------+        |     +-----------------+    |
// |               |       |                |        |     |                 |    |
// | isQ Simulator +-------+ isQ QIR Facade |<-------+     | isQ QIR Shim    |    |
// |               |       |                |Dynamic |     |                 |    |
// +---------------+       +----------------+Linking |     +-----------------+    |
//                                                   |         ^                  |
//                                                   |         |Link against      |
//                                                   |         |                  |
//                                                   |     +---+-------------+    |
//                                                   |     |                 |    |
//                                                   |     | Quantum program |    |
//                                                   |     | in QIR          |    |
//                                                   |     |                 |    |
//                                                   |     +-----------------+    |
//                                                   |                            |
//                                                   |                            |
//                                                   |      Shared library        |
//                                                   +----------------------------+


pub type QIRInt = i64;
pub type QIRDouble = f64;
pub type QIRBool = bool;
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum QIRPauli{
    I=0, X=1, Y=2, Z=3
}
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct QIRRange{
    start: QIRInt,
    step: QIRInt,
    end: QIRInt
}
// We pack QIRResult directly into a pointer.
// TODO: do we need async measurement support?
// We are not real devices, but GPU has its own asynchronicity.
pub type QIRResultPtr = *const ();
pub const QIR_RESULT_ZERO: QIRResultPtr = 1 as QIRResultPtr;
pub const QIR_RESULT_ONE: QIRResultPtr = 2 as QIRResultPtr;



struct QIRContext<T: QDevice>{
    device: T,
    
}

impl<T: QDevice> QIRContext<T>{

}