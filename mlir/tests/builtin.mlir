module @isq_builtin {
    isq.declare_qop @measure : [1]()->i1
    isq.defgate @hadamard : !isq.gate<1, hermitian>
    isq.defgate @cnot : !isq.gate<2, hermitian>
}