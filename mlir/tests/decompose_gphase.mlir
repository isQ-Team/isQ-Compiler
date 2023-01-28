isq.defgate @GPhase(f64) {definition = [{type = "qir", value = @__quantum__qis__gphase}]} : !isq.gate<0>
func private @quantum__qis__gphase(f64)
func @foo(%a: !isq.qstate, %b: !isq.qstate, %c: !isq.qstate, %d: !isq.qstate, %e: !isq.qstate)->(!isq.qstate, !isq.qstate, !isq.qstate, !isq.qstate, !isq.qstate){
    %theta = arith.constant 1000.0 : f64
    %GP = isq.use @GPhase(%theta) : (f64)->!isq.gate<0>
    %U = isq.decorate (%GP: !isq.gate<0>) {ctrl = [true, false, true, false, true], adjoint = true} : !isq.gate<5>
    %f:5 = isq.apply %U(%a, %b, %c, %d, %e) : !isq.gate<5>
    return %f#0, %f#1, %f#2, %f#3, %f#4 : !isq.qstate, !isq.qstate, !isq.qstate, !isq.qstate, !isq.qstate

}
