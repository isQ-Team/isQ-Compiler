module @isq_builtin {
    isq.declare_qop @measure : [1]()->i1
    isq.defgate @hadamard : !isq.gate<1, hermitian>
    isq.defgate @cnot : !isq.gate<2, hermitian>
    isq.defgate @cz : !isq.gate<2, hermitian, symmetric>
}

func @coin()->i1{
    %zero = arith.constant 0: index
    %coinspace = memref.alloca() : memref<1x!isq.qstate>
    %0 = isq.use @isq_builtin::@hadamard : !isq.gate<1, hermitian>
    %down = isq.downgrade (%0: !isq.gate<1, hermitian>) : !isq.gate<1>
    %1 = memref.load %coinspace[%zero] : memref<1x!isq.qstate>
    %3 = isq.apply %down (%1) : !isq.gate<1>
    %4, %outcome = isq.call_qop @isq_builtin::@measure(%3) : [1]()->i1
    return %outcome : i1
}

func @two_dimensional_coins(%coin: memref<?x?x!isq.qstate>, %results: memref<?x?xi1>)->(){
    %c0 = arith.constant 0 : index
    %x = memref.dim %coin, %c0 : memref<?x?x!isq.qstate>
    %c1 = arith.constant 1 : index
    %y = memref.dim %coin, %c1 : memref<?x?x!isq.qstate>
    affine.for %i = 0 to %x step 1{
        affine.for %j = 0 to %y step 1{
            %q = affine.load %coin[%i, %j]: memref<?x?x!isq.qstate>
            %hadamard = isq.use @isq_builtin::@hadamard : !isq.gate<1, hermitian>
            %q1 = isq.apply %hadamard(%q): !isq.gate<1, hermitian>
            %q2, %outcome = isq.call_qop @isq_builtin::@measure(%q1) : [1]()->i1
            affine.store %q2, %coin[%i, %j]: memref<?x?x!isq.qstate>
            affine.store %outcome, %results[%i, %j]: memref<?x?xi1>
        }
    }

    return
}
func @cse_test(%q: !isq.qstate)->!isq.qstate{
    %hadamard_1 = isq.use @isq_builtin::@hadamard : !isq.gate<1, hermitian>
    %hadamard_2 = isq.use @isq_builtin::@hadamard : !isq.gate<1, hermitian>
    %hadamard_3 = isq.use @isq_builtin::@hadamard : !isq.gate<1, hermitian>
    %hadamard_4 = isq.use @isq_builtin::@hadamard : !isq.gate<1, hermitian>
    %q1 = isq.apply %hadamard_1(%q): !isq.gate<1, hermitian>
    %q2 = isq.apply %hadamard_2(%q1): !isq.gate<1, hermitian>
    %q3 = isq.apply %hadamard_3(%q2): !isq.gate<1, hermitian>
    %q4 = isq.apply %hadamard_4(%q3): !isq.gate<1, hermitian>
    return %q4: !isq.qstate
}
