pub type QIRInt = i64;
pub type QIRDouble = f64;
pub type QIRBool = bool;
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum QIRPauli {
    PauliI = 0,
    PauliX = 1,
    PauliY = 3,
    PauliZ = 2,
}
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct QIRRange {
    pub start: QIRInt,
    pub step: QIRInt,
    pub end: QIRInt,
}

impl QIRRange {
    pub fn iter(self) -> impl Iterator<Item = usize> {
        if self.step == 0 {
            panic!("QIRRange::iter: step must be non-zero ({:?})", self);
        }
        let mut i = self.start;
        core::iter::from_fn(move || {
            let ret = i;
            i += self.step;
            if (self.step > 0 && ret > self.end) || (self.step < 0 && ret < self.end) {
                None
            } else {
                Some(ret as usize)
            }
        })
    }
}

// We pack QIRResult directly into a pointer.
// TODO: do we need async measurement support?
// We are not real devices, but GPU has its own asynchronicity.
pub type QIRResultPtr = *const ();
pub const QIR_RESULT_NULL: QIRResultPtr = core::ptr::null();
pub const QIR_RESULT_ZERO: QIRResultPtr = 1 as QIRResultPtr;
pub const QIR_RESULT_ONE: QIRResultPtr = 2 as QIRResultPtr;

pub type QIRResourcePtr = usize;
pub type QIRQubit = usize;

pub use super::array::QIRArray;
pub use super::bigint::QIRBigInt;
pub use super::callable::QIRCallable;
pub use super::string::QIRString;
pub use super::tuple::{QIRTuple, QTupleContent};
