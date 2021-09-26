module @isq_builtin {
    isq.declare_qop @measure : [1]()->i1
    isq.defgate @hadamard : !isq.gate<1, hermitian>
    isq.defgate @cnot : !isq.gate<2, hermitian>
}
func @bell()->(i1, i1){
    %qarr = memref.alloca() : memref<2x!isq.qstate>
    %i0 = constant 0 : index
    %q0 = affine.load %qarr[%i0]: memref<2x!isq.qstate>
    %i1 = constant 1 : index
    %q1 = affine.load %qarr[%i1] : memref<2x!isq.qstate>
    %g1 = isq.use @isq_builtin::@hadamard : !isq.gate<1, hermitian>
    %g2 = isq.use @isq_builtin::@cnot : !isq.gate<2, hermitian>
    %q2 = isq.apply %g1(%q0) : !isq.gate<1, hermitian>
    %q3, %q4 = isq.apply %g2(%q2, %q1) : !isq.gate<2, hermitian>
    %q5, %a = isq.call_qop @isq_builtin::@measure(%q3) : (!isq.qstate)->(!isq.qstate, i1)
    %q6, %b = isq.call_qop @isq_builtin::@measure(%q4) : (!isq.qstate)->(!isq.qstate, i1)
    affine.store %q5, %qarr[%i0] : memref<2x!isq.qstate>
    affine.store %q6, %qarr[%i1] : memref<2x!isq.qstate>
    return %a, %b : i1, i1
}