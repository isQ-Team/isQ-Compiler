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
