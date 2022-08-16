func @decorate_elimination(%a: !isq.gate<1>)->(!isq.gate<1>){
    %b = isq.decorate(%a: !isq.gate<1>) {ctrl = [], adjoint = true}: !isq.gate<1>
    %c = isq.decorate(%b: !isq.gate<1>) {ctrl = [], adjoint = true} : !isq.gate<1>
    return %c: !isq.gate<1>
}
func @decorate_elimination_hermitian(%a: !isq.gate<1, hermitian>)->(!isq.gate<1, hermitian>){
    %b = isq.decorate(%a: !isq.gate<1, hermitian>) {ctrl = [], adjoint = true}: !isq.gate<1, hermitian>
    return %b: !isq.gate<1, hermitian>
}