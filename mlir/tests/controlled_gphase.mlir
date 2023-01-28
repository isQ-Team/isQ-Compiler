isq.defgate @U {definition = [{type = "decomposition", value = @U_decomp}]} : !isq.gate<1>
isq.defgate @MyGPhase(f64) : !isq.gate<0>
func @U_decomp(%q: !isq.qstate)->!isq.qstate{
    %x = arith.constant 1.14 : f64
    %g = isq.use @MyGPhase(%x) : (f64)->!isq.gate<0>
    isq.apply_gphase %g : !isq.gate<0>
    return %q: !isq.qstate
}


func @V(%q1: !isq.qstate, %q2: !isq.qstate)->(!isq.qstate, !isq.qstate){
    %U = isq.use @U : !isq.gate<1>
    %CU = isq.decorate (%U : !isq.gate<1>) {adjoint=false, ctrl=[true]} : !isq.gate<2>
    %q3, %q4 = isq.apply %CU(%q1, %q2) : !isq.gate<2>
    return %q3, %q4 : !isq.qstate, !isq.qstate
}
