func @useless_global_phase(%P : !isq.gate<1, diagonal>, %a: !isq.qstate)-> !isq.qstate{
    %Q = memref.alloc() : memref<1x!isq.qstate>
    %zero = arith.constant 0 : index
    %q1 = affine.load %Q[%zero] : memref<1x!isq.qstate>
    %q2 = isq.apply %P(%q1) : !isq.gate<1, diagonal>
    affine.store %q2, %Q[%zero] : memref<1x!isq.qstate>
    memref.dealloc %Q : memref<1x!isq.qstate>
    return %a: !isq.qstate
}