use alloc::vec::Vec;
use num_bigint::BigInt;

// Immutable QBigInt.
pub struct QBigInt {
    body: BigInt,
    // TODO: QIR Spec does not make clear about the lifetime of get_data.
    // Even the Q# runtime failed to provide a good implementation.
    // We hereby requires the bigint to be immutable: any writing into the last raw data results in undefined behaviour.
    last_raw_data: Vec<u8>,
}

impl QBigInt {
    fn from_bigint(a: BigInt) -> Self {
        let mut s = QBigInt {
            body: a,
            last_raw_data: Vec::new(),
        };
        s.fillin_raw_data();
        s
    }
    pub fn from_i64(a: i64) -> Self {
        Self::from_bigint(BigInt::from(a))
    }
    pub fn from_byte_array(a: &[u8]) -> Self {
        Self::from_bigint(BigInt::from_signed_bytes_be(a))
    }
    fn fillin_raw_data(&mut self) {
        self.last_raw_data = self.body.to_signed_bytes_be();
    }
    pub fn get_raw(&self) -> &[u8] {
        &self.last_raw_data
    }
    pub fn get_bigint(&self) -> &BigInt {
        &self.body
    }
}

pub type QIRBigint = QBigInt;
