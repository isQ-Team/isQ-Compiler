#invsqrt2 = #isq.complex<0.7071067811865475, 0.0>
#ninvsqrt2 = #isq.complex<-0.7071067811865475, 0.0>
module @isq_builtin {
    
    isq.declare_qop @measure : [1]()->i1
    isq.declare_qop @reset : [1]()->i1
    isq.defgate @hadamard {definition = [{type = "unitary", value = [
        [#invsqrt2,#invsqrt2],
        [#invsqrt2,#ninvsqrt2]]}]}: !isq.gate<1, hermitian>
    isq.defgate @swap {definition = [{type="unitary", value = [
        [#isq.complex<1.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>],
        [#isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<1.0, 0.0>, #isq.complex<0.0, 0.0>],
        [#isq.complex<0.0, 0.0>, #isq.complex<1.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>],
        [#isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<1.0, 0.0>]]},
        {type="decomposition", value = @swap_impl}]} : !isq.gate<2, hermitian, symmetric>
    isq.defgate @cnot : !isq.gate<2, hermitian>
    isq.defgate @cz : !isq.gate<2, hermitian, symmetric>
    func @swap_impl(%a: !isq.qstate, %b: !isq.qstate)->(!isq.qstate, !isq.qstate){
        %cnot = isq.use @cnot : !isq.gate<2, hermitian>
        %a1, %b1 = isq.apply %cnot(%a, %b) : !isq.gate<2, hermitian>
        %b2, %a2 = isq.apply %cnot(%b1, %a1) : !isq.gate<2, hermitian>
        %a3, %b3 = isq.apply %cnot(%a2, %b2) : !isq.gate<2, hermitian>
        return %a3, %b3: !isq.qstate, !isq.qstate
    }
}