func @double_control(%a: !isq.gate<1>)->(!isq.gate<3>, !isq.gate<4>){
    %b = isq.decorate(%a: !isq.gate<1>) {ctrl = [true], adjoint = true}: !isq.gate<2>
    %c = isq.decorate(%b: !isq.gate<2>) {ctrl = [false], adjoint = true} : !isq.gate<3>
    %d = isq.decorate(%c: !isq.gate<3>) {ctrl = [true], adjoint = true} : !isq.gate<4>
    return %c, %d : !isq.gate<3>, !isq.gate<4>
}