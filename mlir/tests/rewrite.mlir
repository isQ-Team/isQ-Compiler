module @isq_builtin {
    isq.declare_qop @measure : [1]()->i1
    isq.defgate @hadamard : !isq.gate<1, hermitian>
    isq.defgate @cnot : !isq.gate<2, hermitian>
    isq.defgate @cz : !isq.gate<2, hermitian, symmetric>
}
func @rewrite_test(%q: !isq.qstate)->!isq.qstate{
    %hadamard = isq.use @isq_builtin::@hadamard : !isq.gate<1, hermitian>
    %q1 = isq.apply %hadamard(%q): !isq.gate<1, hermitian>
    %q2 = isq.apply %hadamard(%q1): !isq.gate<1, hermitian>
    return %q2: !isq.qstate
}
func @unidirectional_cnot_should_cancel(%p1: !isq.qstate, %q1: !isq.qstate)->(!isq.qstate, !isq.qstate){
    %cnot = isq.use @isq_builtin::@cnot : !isq.gate<2, hermitian>
    %p2, %q2 = isq.apply %cnot(%p1, %q1): !isq.gate<2, hermitian>
    %p3, %q3 = isq.apply %cnot(%p2, %q2): !isq.gate<2, hermitian>
    return %p3, %q3: !isq.qstate, !isq.qstate
}
func @bidirectional_cnot_should_not_cancel(%p1: !isq.qstate, %q1: !isq.qstate)->(!isq.qstate, !isq.qstate){
    %cnot = isq.use @isq_builtin::@cnot : !isq.gate<2, hermitian>
    %p2, %q2 = isq.apply %cnot(%p1, %q1): !isq.gate<2, hermitian>
    %p3, %q3 = isq.apply %cnot(%q2, %p2): !isq.gate<2, hermitian>
    %p4, %q4 = isq.apply %cnot(%q3, %p3): !isq.gate<2, hermitian>
    return %p4, %q4: !isq.qstate, !isq.qstate
}
func @unidirectional_cz_should_cancel(%p1: !isq.qstate, %q1: !isq.qstate)->(!isq.qstate, !isq.qstate){
    %cz = isq.use @isq_builtin::@cz : !isq.gate<2, hermitian, symmetric>
    %p2, %q2 = isq.apply %cz(%p1, %q1): !isq.gate<2, hermitian, symmetric>
    %p3, %q3 = isq.apply %cz(%p2, %q2): !isq.gate<2, hermitian, symmetric>
    return %p3, %q3: !isq.qstate, !isq.qstate
}
func @biidirectional_cz_should_cancel(%p1: !isq.qstate, %q1: !isq.qstate)->(!isq.qstate, !isq.qstate){
    %cz = isq.use @isq_builtin::@cz : !isq.gate<2, hermitian, symmetric>
    %p2, %q2 = isq.apply %cz(%p1, %q1): !isq.gate<2, hermitian, symmetric>
    %p3, %q3 = isq.apply %cz(%q2, %p2): !isq.gate<2, hermitian, symmetric>
    return %p3, %q3: !isq.qstate, !isq.qstate
}
func @cancellation_with_cse(%q: !isq.qstate)->!isq.qstate{
    %hadamard_1 = isq.use @isq_builtin::@hadamard : !isq.gate<1, hermitian>
    %hadamard_2 = isq.use @isq_builtin::@hadamard : !isq.gate<1, hermitian>
    %q1 = isq.apply %hadamard_1(%q): !isq.gate<1, hermitian>
    %q2 = isq.apply %hadamard_2(%q1): !isq.gate<1, hermitian>
    return %q2: !isq.qstate
}
func @two_dimensional_do_nothing(%coin: memref<?x?x!isq.qstate>)->(){
    
    %c0 = constant 0 : index
    %x = memref.dim %coin, %c0 : memref<?x?x!isq.qstate>
    %c1 = constant 1 : index
    %y = memref.dim %coin, %c1 : memref<?x?x!isq.qstate>
    affine.for %i = 0 to %x step 1{
        affine.for %j = 1 to %y step 1{
            %q = affine.load %coin[%i, %j]: memref<?x?x!isq.qstate>
            %hadamard = isq.use @isq_builtin::@hadamard : !isq.gate<1, hermitian>
            %q1 = isq.apply %hadamard(%q): !isq.gate<1, hermitian>
            %q2 = isq.apply %hadamard(%q1): !isq.gate<1, hermitian>
            affine.store %q2, %coin[%i, %j]: memref<?x?x!isq.qstate>
        }
    }
    return
}