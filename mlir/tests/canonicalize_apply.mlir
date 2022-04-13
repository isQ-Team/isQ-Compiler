func @interleave_apply(%A: !isq.gate<4>, %B: !isq.gate<2, symmetric>, %q1: !isq.qstate, %q2: !isq.qstate, %q3: !isq.qstate, %q4: !isq.qstate)->(!isq.qstate,!isq.qstate,!isq.qstate,!isq.qstate){
    %q5, %q6, %q7, %q8 = isq.apply %A(%q1, %q2, %q3, %q4): !isq.gate<4>
    %q12, %q9 = isq.apply %B(%q8, %q5) : !isq.gate<2, symmetric>
    %q11, %q10 = isq.apply %B(%q7, %q6) : !isq.gate<2, symmetric>
    return %q9, %q10, %q11, %q12 : !isq.qstate,!isq.qstate,!isq.qstate,!isq.qstate
}


func @symmetry_cancel(%A: !isq.gate<3, hermitian, symmetric>, %q1: !isq.qstate, %q2: !isq.qstate, %q3: !isq.qstate)->(!isq.qstate,!isq.qstate,!isq.qstate){
    %p1:3 = isq.apply %A(%q1, %q2, %q3) : !isq.gate<3, hermitian, symmetric>
    %p2:3 = isq.apply %A(%p1#0, %p1#2, %p1#1): !isq.gate<3, hermitian, symmetric>
    %p3:3 = isq.apply %A(%p2#1, %p2#2, %p2#0): !isq.gate<3, hermitian, symmetric>
    %p4:3 = isq.apply %A(%p3#1, %p3#0, %p3#2): !isq.gate<3, hermitian, symmetric>
    %p5:3 = isq.apply %A(%p4#2, %p4#0, %p4#1): !isq.gate<3, hermitian, symmetric>
    %p6:3 = isq.apply %A(%p5#2, %p5#1, %p5#0): !isq.gate<3, hermitian, symmetric>
    return %p6#2, %p6#1, %p6#0 : !isq.qstate,!isq.qstate,!isq.qstate
}

func @symmetry_cancel_adj(%A: !isq.gate<3, symmetric>, %q1: !isq.qstate, %q2: !isq.qstate, %q3: !isq.qstate)->(!isq.qstate,!isq.qstate,!isq.qstate){
    %Aadj = isq.decorate(%A: !isq.gate<3, symmetric>) {ctrl=[], adjoint = true} : !isq.gate<3, symmetric>
    %p1:3 = isq.apply %A(%q1, %q2, %q3) : !isq.gate<3, symmetric>
    %p2:3 = isq.apply %Aadj(%p1#0, %p1#2, %p1#1): !isq.gate<3, symmetric>
    %p3:3 = isq.apply %Aadj(%p2#1, %p2#2, %p2#0): !isq.gate<3, symmetric>
    %p4:3 = isq.apply %A(%p3#1, %p3#0, %p3#2): !isq.gate<3, symmetric>
    %p5:3 = isq.apply %A(%p4#2, %p4#0, %p4#1): !isq.gate<3, symmetric>
    %p6:3 = isq.apply %Aadj(%p5#2, %p5#1, %p5#0): !isq.gate<3, symmetric>
    return %p6#2, %p6#1, %p6#0 : !isq.qstate,!isq.qstate,!isq.qstate
}
