func @apply_downgrade(%G: !isq.gate<1, hermitian>, %q: !isq.qstate)->!isq.qstate{
    %Gd = isq.downgrade(%G: !isq.gate<1, hermitian>) : !isq.gate<1>
    %q2 = isq.apply %Gd(%q) :!isq.gate<1>
    return %q2: !isq.qstate
}
func @noop_downgrade(%G : !isq.gate<1, hermitian>)->!isq.gate<1, hermitian>{
    %Gd = isq.downgrade(%G: !isq.gate<1, hermitian>) : !isq.gate<1, hermitian>
    return %Gd: !isq.gate<1, hermitian>
}

func @merge_downgrade(%G : !isq.gate<1, hermitian, diagonal>)->!isq.gate<1>{
    %G1 = isq.downgrade(%G: !isq.gate<1, hermitian, diagonal>) : !isq.gate<1, hermitian>
    %G2 = isq.downgrade(%G1: !isq.gate<1, hermitian>) : !isq.gate<1>
    return %G2: !isq.gate<1>
}