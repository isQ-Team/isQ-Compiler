#invsqrt2 = #isq.complex<0.7071067811865475, 0.0>
#ninvsqrt2 = #isq.complex<-0.7071067811865475, 0.0>

isq.defgate @hadamard {definition = [{type = "unitary", value = [
    [#invsqrt2,#invsqrt2],
    [#invsqrt2,#ninvsqrt2]]}]}: !isq.gate<1>

func @c_h(%a: !isq.qstate, %b: !isq.qstate)->(!isq.qstate, !isq.qstate){
    %H = isq.use @hadamard : !isq.gate<1>
    %CH = isq.decorate(%H: !isq.gate<1>) {ctrl = [true], adjoint=false} : !isq.gate<2>
    %a1, %b1 = isq.apply %CH(%a, %b) : !isq.gate<2>
    return %a1, %b1: !isq.qstate, !isq.qstate
}

func @c_nc_h(%a: !isq.qstate, %b: !isq.qstate, %c: !isq.qstate)->(!isq.qstate, !isq.qstate, !isq.qstate){
    %H = isq.use @hadamard : !isq.gate<1>
    %CH = isq.decorate(%H: !isq.gate<1>) {ctrl = [false], adjoint=false} : !isq.gate<2>
    %CCH = isq.decorate(%CH: !isq.gate<2>) {ctrl = [true], adjoint=true} : !isq.gate<3>
    %a1, %b1, %c1 = isq.apply %CCH(%a, %b, %c) : !isq.gate<3>
    return %a1, %b1, %c1: !isq.qstate, !isq.qstate, !isq.qstate
}