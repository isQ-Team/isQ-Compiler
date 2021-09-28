#invsqrt2 = #isq.complex<0.7071067811865475, 0.0>
#ninvsqrt2 = #isq.complex<-0.7071067811865475, 0.0>
module @isq_builtin {
    isq.declare_qop @measure : [1]()->i1
    isq.declare_qop @reset : [1]()->i1
    isq.defgate @hadamard {definition = [{type = "unitary", value = [
        [#invsqrt2,#invsqrt2],
        [#invsqrt2,#ninvsqrt2]]}]}: !isq.gate<1, hermitian>
    isq.defgate @cnot : !isq.gate<2, hermitian>
    isq.defgate @cz : !isq.gate<2, hermitian, symmetric>
}