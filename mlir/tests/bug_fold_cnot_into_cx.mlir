func @my_cnot(%a: !isq.qstate, %b: !isq.qstate, %c: !isq.qstate)->(!isq.qstate, !isq.qstate, !isq.qstate){
    %g = isq.use @__isq__builtin__cnot : !isq.gate<2>
    %g2 = isq.decorate(%g: !isq.gate<2>) {ctrl=[true], adjoint = false} : !isq.gate<3>
    %d, %e, %f = isq.apply %g2(%a, %b, %c) : !isq.gate<3>
    return %d, %e, %f: !isq.qstate, !isq.qstate, !isq.qstate
}
isq.defgate @__isq__builtin__cnot {definition = [{type = "qir", value = @__quantum__qis__cnot}]} : !isq.gate<2>