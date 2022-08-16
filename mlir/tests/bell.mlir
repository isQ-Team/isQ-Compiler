isq.declare_qop @__isq__builtin__measure : [1]()->i1
isq.declare_qop @__isq__builtin__reset : [1]()->()
isq.declare_qop @__isq__builtin__print_int : [0](index)->()
isq.declare_qop @__isq__builtin__print_double : [0](f64)->()
isq.defgate @__isq__builtin__u3(f64, f64, f64) {definition = [{type = "qir", value = "__quantum__qis__u3"}]} : !isq.gate<1>
isq.defgate @__isq__builtin__cnot {definition = [{type = "qir", value = "__quantum__qis__cnot"}]} : !isq.gate<2>
func private @__quantum__qis__u3(f64, f64, f64, !isq.qir.qubit)
func private @__quantum__qis__cnot(!isq.qir.qubit, !isq.qir.qubit)
func @bell()->(i1, i1){
    %qarr = memref.alloca() : memref<2x!isq.qstate>
    %i0 = arith.constant 0 : index
    %q0 = affine.load %qarr[%i0]: memref<2x!isq.qstate>
    %i1 = arith.constant 1 : index
    %q1 = affine.load %qarr[%i1] : memref<2x!isq.qstate>
    %g1 = isq.use @isq_builtin::@hadamard : !isq.gate<1, hermitian>
    %g2 = isq.use @isq_builtin::@cnot : !isq.gate<2, hermitian>
    %q2 = isq.apply %g1(%q0) : !isq.gate<1, hermitian>
    %q3, %q4 = isq.apply %g2(%q2, %q1) : !isq.gate<2, hermitian>
    %q5, %a = isq.call_qop @isq_builtin::@measure(%q3) : [1]()->i1
    %q6, %b = isq.call_qop @isq_builtin::@measure(%q4) : [1]()->i1
    affine.store %q5, %qarr[%i0] : memref<2x!isq.qstate>
    affine.store %q6, %qarr[%i1] : memref<2x!isq.qstate>
    return %a, %b : i1, i1
}