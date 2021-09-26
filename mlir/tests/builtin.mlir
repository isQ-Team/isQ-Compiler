module @isq_builtin {
    isq.declare_qop @measure : [1]()->i1
    isq.declare_qop @reset : [1]()->i1
    isq.defgate @hadamard {definition = []}: !isq.gate<1, hermitian>
    isq.defgate @cnot : !isq.gate<2, hermitian>
    isq.defgate @cz : !isq.gate<2, hermitian, symmetric>
}