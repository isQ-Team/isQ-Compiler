#single_qubit_measurement = !isq.qop<(!isq.qstate)->(!isq.qstate, i1)>
module{
    func @coin()->i1{
        %zero = std.constant 0: index
        %coinspace = memref.alloca() : memref<1x!isq.qstate>
        %0 = isq.gate {name = "hadamard", gate_type = !isq.gate<1, hermitian>} : !isq.gate<1, hermitian>
        %down = isq.downgrade (%0: !isq.gate<1, hermitian>) : !isq.gate<1>
        %meas = isq.declare_op {name = "computational_basis_measurement", op_type = #single_qubit_measurement}: !isq.qop<(!isq.qstate)->(!isq.qstate, i1)>
        %1 = memref.load %coinspace[%zero] : memref<1x!isq.qstate>
        %2 = isq.use_gate %down : !isq.gate<1>
        %3 = isq.apply %2 (%1) : !isq.qop<(!isq.qstate)->!isq.qstate>
        %4, %outcome = isq.apply %meas (%3) : !isq.qop<(!isq.qstate)->(!isq.qstate, i1)>
        return %outcome : i1
    }

    func @two_dimensional_coins(%coin: memref<?x?x!isq.qstate>, %results: memref<?x?xi1>)->(){
        
        %c0 = constant 0 : index
        %x = memref.dim %coin, %c0 : memref<?x?x!isq.qstate>
        %c1 = constant 1 : index
        %y = memref.dim %coin, %c1 : memref<?x?x!isq.qstate>
        affine.for %i = 0 to %x step 1{
            affine.for %j = 1 to %y step 1{
                %q = affine.load %coin[%i, %j]: memref<?x?x!isq.qstate>
                %hadamard = isq.gate {name = "hadamard", gate_type = !isq.gate<1, hermitian>} : !isq.gate<1, hermitian>
                %hadamard_op = isq.use_gate %hadamard : !isq.gate<1, hermitian>
                %meas = isq.declare_op {name = "computational_basis_measurement", op_type = #single_qubit_measurement}: !isq.qop<(!isq.qstate)->(!isq.qstate, i1)>
                %q1 = isq.apply %hadamard_op(%q): !isq.qop<(!isq.qstate)->!isq.qstate>
                %q2, %outcome = isq.apply %meas (%q1) : !isq.qop<(!isq.qstate)->(!isq.qstate, i1)>
                affine.store %q2, %coin[%i, %j]: memref<?x?x!isq.qstate>
                affine.store %outcome, %results[%i, %j]: memref<?x?xi1>
            }
        }
        return
    }
    func @cse_test(%q: !isq.qstate)->!isq.qstate{
        %hadamard_1 = isq.gate {name = "hadamard", gate_type = !isq.gate<1, hermitian>} : !isq.gate<1, hermitian>
        %hadamard_1_op = isq.use_gate %hadamard_1 : !isq.gate<1, hermitian>
        %hadamard_2 = isq.gate {name = "hadamard", gate_type = !isq.gate<1, hermitian>} : !isq.gate<1, hermitian>
        %hadamard_2_op = isq.use_gate %hadamard_2 : !isq.gate<1, hermitian>
        %hadamard_3 = isq.gate {name = "hadamard", gate_type = !isq.gate<1, hermitian>} : !isq.gate<1, hermitian>
        %hadamard_3_op = isq.use_gate %hadamard_3 : !isq.gate<1, hermitian>
        %hadamard_4 = isq.gate {name = "Z", gate_type = !isq.gate<1, hermitian>} : !isq.gate<1, hermitian>
        %hadamard_4_op = isq.use_gate %hadamard_4 : !isq.gate<1, hermitian>
        %q1 = isq.apply %hadamard_1_op(%q): !isq.qop<(!isq.qstate)->!isq.qstate>
        %q2 = isq.apply %hadamard_2_op(%q1): !isq.qop<(!isq.qstate)->!isq.qstate>
        %q3 = isq.apply %hadamard_3_op(%q2): !isq.qop<(!isq.qstate)->!isq.qstate>
        %q4 = isq.apply %hadamard_4_op(%q3): !isq.qop<(!isq.qstate)->!isq.qstate>
        return %q4: !isq.qstate
    }
}