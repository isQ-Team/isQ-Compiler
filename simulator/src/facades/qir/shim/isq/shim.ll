%Range = type { i64, i64, i64 }
%Tuple = type opaque
%Qubit = type opaque
%Result = type opaque
%Array = type opaque
%Callable = type opaque
%BigInt = type opaque
%Pauli = type i2
%String = type opaque
define void @__quantum__qis__u3 (double %x0, double %x1, double %x2, %Qubit* %x3) alwaysinline {
entry:
    %x4 = bitcast double %x0 to double
    %x5 = bitcast double %x1 to double
    %x6 = bitcast double %x2 to double
    %x7 = bitcast %Qubit* %x3 to i8*
    call void @__isq__qir__shim__qis__u3(double %x4, double %x5, double %x6, i8* %x7)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__u3(double, double, double, i8*)
define void @__quantum__qis__rz__body (double %x0, %Qubit* %x1) alwaysinline {
entry:
    %x2 = bitcast double %x0 to double
    %x3 = bitcast %Qubit* %x1 to i8*
    call void @__isq__qir__shim__qis__rz__body(double %x2, i8* %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__rz__body(double, i8*)
define void @__quantum__qis__rx__body (double %x0, %Qubit* %x1) alwaysinline {
entry:
    %x2 = bitcast double %x0 to double
    %x3 = bitcast %Qubit* %x1 to i8*
    call void @__isq__qir__shim__qis__rx__body(double %x2, i8* %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__rx__body(double, i8*)
define void @__quantum__qis__ry__body (double %x0, %Qubit* %x1) alwaysinline {
entry:
    %x2 = bitcast double %x0 to double
    %x3 = bitcast %Qubit* %x1 to i8*
    call void @__isq__qir__shim__qis__ry__body(double %x2, i8* %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__ry__body(double, i8*)
define void @__quantum__qis__rxp__body (i8* %x0, i64 %x1, i64 %x2, %Qubit* %x3) alwaysinline {
entry:
    %x4 = bitcast %Qubit* %x3 to i8*
    call void @__isq__qir__shim__qis__rxp__body(i8* %x0, i64 %x1, i64 %x2, i8* %x4)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__rxp__body(i8*, i64, i64, i8*)

define void @__quantum__qis__ryp__body (i8* %x0, i64 %x1, i64 %x2, %Qubit* %x3) alwaysinline {
entry:
    %x4 = bitcast %Qubit* %x3 to i8*
    call void @__isq__qir__shim__qis__ryp__body(i8* %x0, i64 %x1, i64 %x2, i8* %x4)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__ryp__body(i8*, i64, i64, i8*)

define void @__quantum__qis__rzp__body (i8* %x0, i64 %x1, i64 %x2, %Qubit* %x3) alwaysinline {
entry:
    %x4 = bitcast %Qubit* %x3 to i8*
    call void @__isq__qir__shim__qis__rzp__body(i8* %x0, i64 %x1, i64 %x2, i8* %x4)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__rzp__body(i8*, i64, i64, i8*)

define void @__quantum__qis__x2p (%Qubit* %x0) alwaysinline {
entry:
    %x1 = bitcast %Qubit* %x0 to i8*
    call void @__isq__qir__shim__qis__x2p(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__x2p(i8*)
define void @__quantum__qis__x2m (%Qubit* %x0) alwaysinline {
entry:
    %x1 = bitcast %Qubit* %x0 to i8*
    call void @__isq__qir__shim__qis__x2m(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__x2m(i8*)
define void @__quantum__qis__y2p (%Qubit* %x0) alwaysinline {
entry:
    %x1 = bitcast %Qubit* %x0 to i8*
    call void @__isq__qir__shim__qis__y2p(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__y2p(i8*)
define void @__quantum__qis__y2m (%Qubit* %x0) alwaysinline {
entry:
    %x1 = bitcast %Qubit* %x0 to i8*
    call void @__isq__qir__shim__qis__y2m(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__y2m(i8*)
define void @__quantum__qis__gphase (double %x0) alwaysinline {
entry:
    %x1 = bitcast double %x0 to double
    call void @__isq__qir__shim__qis__gphase(double %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__gphase(double)
define void @__quantum__qis__cnot (%Qubit* %x0, %Qubit* %x1) alwaysinline {
entry:
    %x2 = bitcast %Qubit* %x0 to i8*
    %x3 = bitcast %Qubit* %x1 to i8*
    call void @__isq__qir__shim__qis__cnot(i8* %x2, i8* %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__cnot(i8*, i8*)
define void @__quantum__qis__cz (%Qubit* %x0, %Qubit* %x1) alwaysinline {
entry:
    %x2 = bitcast %Qubit* %x0 to i8*
    %x3 = bitcast %Qubit* %x1 to i8*
    call void @__isq__qir__shim__qis__cz(i8* %x2, i8* %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__cz(i8*, i8*)
define %Result* @__quantum__qis__measure (%Qubit* %x0) alwaysinline {
entry:
    %x1 = bitcast %Qubit* %x0 to i8*
    %x2 = call i8* @__isq__qir__shim__qis__measure(i8* %x1)
    %x3 = bitcast i8* %x2 to %Result*
    ret %Result* %x3
}
declare dllimport i8* @__isq__qir__shim__qis__measure(i8*)
define void @__quantum__qis__reset (%Qubit* %x0) alwaysinline {
entry:
    %x1 = bitcast %Qubit* %x0 to i8*
    call void @__isq__qir__shim__qis__reset(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__reset(i8*)
define void @__quantum__qis__qcis__finalize () alwaysinline {
entry:
    call void @__isq__qir__shim__qis__qcis__finalize()
    ret void
}
declare dllimport void @__isq__qir__shim__qis__qcis__finalize()
define void @__quantum__qis__bp (i64 %x0) alwaysinline {
entry:
    %x1 = bitcast i64 %x0 to i64
    call void @__isq__qir__shim__qis__bp(i64 %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__bp(i64)
define void @__quantum_qis__assert (%Qubit** %qaddr, %Qubit** %qaddr1, i64 %x0, i64 %x1, i64 %x2, double* %saddr, double* %saddr1, i64 %s0, i64 %s1, i64 %s2) alwaysinline {
entry:
    %q1 = bitcast %Qubit** %qaddr to i8**
    %q2 = bitcast %Qubit** %qaddr1 to i8**
    call void @__isq__qir__shim__qis__assert(i8** %q1, i8** %q2, i64 %x0, i64 %x1, i64 %x2, double* %saddr, double* %saddr1, i64 %s0, i64 %s1, i64 %s2)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__assert(i8**, i8**, i64, i64, i64, double*, double*, i64, i64, i64)
define void @__quantum__qis__isq_print_i64 (i64 %x0) alwaysinline {
entry:
    %x1 = bitcast i64 %x0 to i64
    call void @__isq__qir__shim__qis__isq_print_i64(i64 %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__isq_print_i64(i64)
define void @__quantum__qis__isq_print_f64 (double %x0) alwaysinline {
entry:
    %x1 = bitcast double %x0 to double
    call void @__isq__qir__shim__qis__isq_print_f64(double %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__isq_print_f64(double)
declare dllimport void @__isq__qir__shim__qmpi__csend(i64, i64, i1)
declare dllimport i1 @__isq__qir__shim__qmpi__crecv(i64, i64)
define void @__quantum__qmpi__create__epr (i64 %peer, i64 %tag, %Qubit* %x0) alwaysinline {
entry:
    %x1 = bitcast %Qubit* %x0 to i8*
    call void @__isq__qir__shim__qmpi__create__epr(i64 %peer, i64 %tag, i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qmpi__create__epr(i64 %peer, i64 %tag, i8* %x1)
declare dllimport i64 @__isq__qir__shim__qmpi__size(i64, i64)
