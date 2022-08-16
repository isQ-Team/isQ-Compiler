module @isq_builtin {
    isq.declare_qop @measure : [1]()->i1
    isq.defgate @hadamard : !isq.gate<1, hermitian>
    isq.defgate @cnot : !isq.gate<2, hermitian>
}

isq.defgate @gates {shape=[10,20]}: !isq.gate<1>

func @scramble_all()->memref<10x20xi1>{
    %qarr = memref.alloca() : memref<10x20x!isq.qstate>
    %garr = isq.use @gates : memref<10x20x!isq.gate<1>>
    %results = memref.alloca() : memref<10x20xi1>
    %c0 = arith.constant 0 : index
    %x = memref.dim %qarr, %c0 : memref<10x20x!isq.qstate>
    %c1 = arith.constant 1 : index
    %y = memref.dim %qarr, %c1 : memref<10x20x!isq.qstate>
    affine.for %i = 0 to %x step 1{
        affine.for %j = 0 to %y step 1 {
            %q = affine.load %qarr[%i, %j]: memref<10x20x!isq.qstate>
            %gate = affine.load %garr[%i, %j]: memref<10x20x!isq.gate<1>>
            %q1 = isq.apply %gate(%q): !isq.gate<1>
            %q2, %outcome = isq.call_qop @isq_builtin::@measure(%q1) : [1]()->i1
            affine.store %q2, %qarr[%i, %j]: memref<10x20x!isq.qstate>
            affine.store %outcome, %results[%i, %j]: memref<10x20xi1>
        }
    }
    return %results : memref<10x20xi1>
}