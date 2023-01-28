isq.defgate @cz : !isq.gate<2, hermitian, symmetric>
func @cz_on_one_qubit(%a: !isq.qstate)->(!isq.qstate, !isq.qstate){
    %cz = isq.use @cz: !isq.gate<2, hermitian, symmetric>
    %b, %c = isq.apply %cz(%a, %a): !isq.gate<2, hermitian, symmetric>
    return %b, %c: !isq.qstate, !isq.qstate
}