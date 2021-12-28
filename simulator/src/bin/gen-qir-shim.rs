// Generates both LLVM and Rust shims simultaneously.
// Since there are many kinds of signatures, we use a DSL to describe each intrinsic function.

// There will be 3 shims: one for QIR spec, one in standard function signature (all arguments and return values being primitive types), one for fetched-back Rust function signature.
/*
QIR: always inline, casting arguments to primitive types, casting return value from primitive type.
*/
enum QIRType{
    Qubit, Array, BigInt, String, Tuple, Callable, Int, Double, Pauli, Result, Unit
}
enum PrimitiveType{
    Pointer, Void, I64, I8, I32
}
struct QIRBuilder{
    
}


fn main(){

}