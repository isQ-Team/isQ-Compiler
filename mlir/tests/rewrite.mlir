func @rewrite_test(%q: !isq.qstate)->!isq.qstate{
    %hadamard = isq.gate {name = "hadamard", gate_type = !isq.gate<1, hermitian>} : !isq.gate<1, hermitian>
    %hadamard_op = isq.use_gate %hadamard : !isq.gate<1, hermitian>
    %q1 = isq.apply %hadamard_op(%q): !isq.qop<(!isq.qstate)->!isq.qstate>
    %q2 = isq.apply %hadamard_op(%q1): !isq.qop<(!isq.qstate)->!isq.qstate>
    return %q2: !isq.qstate
}
func @unidirectional_cnot_should_cancel(%p1: !isq.qstate, %q1: !isq.qstate)->(!isq.qstate, !isq.qstate){
    %cnot = isq.gate {name="cnot", gate_type = !isq.gate<2, hermitian>}: !isq.gate<2, hermitian>
    %cnot_op = isq.use_gate %cnot : !isq.gate<2, hermitian>
    %p2, %q2 = isq.apply %cnot_op(%p1, %q1): !isq.qop<(!isq.qstate, !isq.qstate)->(!isq.qstate, !isq.qstate)>
    %p3, %q3 = isq.apply %cnot_op(%p2, %q2): !isq.qop<(!isq.qstate, !isq.qstate)->(!isq.qstate, !isq.qstate)>
    return %p3, %q3: !isq.qstate, !isq.qstate
}
func @bidirectional_cnot_should_not_cancel(%p1: !isq.qstate, %q1: !isq.qstate)->(!isq.qstate, !isq.qstate){
    %cnot = isq.gate {name="cnot", gate_type = !isq.gate<2, hermitian>}: !isq.gate<2, hermitian>
    %cnot_op = isq.use_gate %cnot : !isq.gate<2, hermitian>
    %p2, %q2 = isq.apply %cnot_op(%p1, %q1): !isq.qop<(!isq.qstate, !isq.qstate)->(!isq.qstate, !isq.qstate)>
    %p3, %q3 = isq.apply %cnot_op(%q2, %p2): !isq.qop<(!isq.qstate, !isq.qstate)->(!isq.qstate, !isq.qstate)>
    return %p3, %q3: !isq.qstate, !isq.qstate
}
func @unidirectional_cz_should_cancel(%p1: !isq.qstate, %q1: !isq.qstate)->(!isq.qstate, !isq.qstate){
    %cz = isq.gate {name="cz", gate_type = !isq.gate<2, hermitian, symmetric>}: !isq.gate<2, hermitian, symmetric>
    %cz_op = isq.use_gate %cz : !isq.gate<2, hermitian, symmetric>
    %p2, %q2 = isq.apply %cz_op(%p1, %q1): !isq.qop<(!isq.qstate, !isq.qstate)->(!isq.qstate, !isq.qstate)>
    %p3, %q3 = isq.apply %cz_op(%p2, %q2): !isq.qop<(!isq.qstate, !isq.qstate)->(!isq.qstate, !isq.qstate)>
    return %p3, %q3: !isq.qstate, !isq.qstate
}
func @biidirectional_cz_should_cancel(%p1: !isq.qstate, %q1: !isq.qstate)->(!isq.qstate, !isq.qstate){
    %cz = isq.gate {name="cz", gate_type = !isq.gate<2, hermitian, symmetric>}: !isq.gate<2, hermitian, symmetric>
    %cz_op = isq.use_gate %cz : !isq.gate<2, hermitian, symmetric>
    %p2, %q2 = isq.apply %cz_op(%p1, %q1): !isq.qop<(!isq.qstate, !isq.qstate)->(!isq.qstate, !isq.qstate)>
    %p3, %q3 = isq.apply %cz_op(%q2, %p2): !isq.qop<(!isq.qstate, !isq.qstate)->(!isq.qstate, !isq.qstate)>
    return %p3, %q3: !isq.qstate, !isq.qstate
}
func @cancellation_with_cse(%q: !isq.qstate)->!isq.qstate{
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