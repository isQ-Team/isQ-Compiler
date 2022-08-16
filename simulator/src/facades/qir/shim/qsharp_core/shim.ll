%Range = type { i64, i64, i64 }
%Tuple = type opaque
%Qubit = type opaque
%Result = type opaque
%Array = type opaque
%Callable = type opaque
%BigInt = type opaque
%Pauli = type i2
%String = type opaque
define void @__quantum__qis__exp__body (%Array* %x0, double %x1, %Array* %x2) alwaysinline {
entry:
    %x3 = bitcast %Array* %x0 to i8*
    %x4 = bitcast double %x1 to double
    %x5 = bitcast %Array* %x2 to i8*
    call void @__isq__qir__shim__qis__exp__body(i8* %x3, double %x4, i8* %x5)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__exp__body(i8*, double, i8*)
define void @__quantum__qis__exp__adj (%Array* %x0, double %x1, %Array* %x2) alwaysinline {
entry:
    %x3 = bitcast %Array* %x0 to i8*
    %x4 = bitcast double %x1 to double
    %x5 = bitcast %Array* %x2 to i8*
    call void @__isq__qir__shim__qis__exp__adj(i8* %x3, double %x4, i8* %x5)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__exp__adj(i8*, double, i8*)
define void @__quantum__qis__exp__ctl (%Array* %x0, %Array* %x1, double %x2, %Array* %x3) alwaysinline {
entry:
    %x4 = bitcast %Array* %x0 to i8*
    %x5 = bitcast %Array* %x1 to i8*
    %x6 = bitcast double %x2 to double
    %x7 = bitcast %Array* %x3 to i8*
    call void @__isq__qir__shim__qis__exp__ctl(i8* %x4, i8* %x5, double %x6, i8* %x7)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__exp__ctl(i8*, i8*, double, i8*)
define void @__quantum__qis__exp__ctladj (%Array* %x0, %Array* %x1, double %x2, %Array* %x3) alwaysinline {
entry:
    %x4 = bitcast %Array* %x0 to i8*
    %x5 = bitcast %Array* %x1 to i8*
    %x6 = bitcast double %x2 to double
    %x7 = bitcast %Array* %x3 to i8*
    call void @__isq__qir__shim__qis__exp__ctladj(i8* %x4, i8* %x5, double %x6, i8* %x7)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__exp__ctladj(i8*, i8*, double, i8*)
define void @__quantum__qis__h__body (%Qubit* %x0) alwaysinline {
entry:
    %x1 = bitcast %Qubit* %x0 to i8*
    call void @__isq__qir__shim__qis__h__body(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__h__body(i8*)
define void @__quantum__qis__h__ctl (%Array* %x0, %Qubit* %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = bitcast %Qubit* %x1 to i8*
    call void @__isq__qir__shim__qis__h__ctl(i8* %x2, i8* %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__h__ctl(i8*, i8*)
define %Result* @__quantum__qis__measure__body (%Array* %x0, %Array* %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = bitcast %Array* %x1 to i8*
    %x4 = call i8* @__isq__qir__shim__qis__measure__body(i8* %x2, i8* %x3)
    %x5 = bitcast i8* %x4 to %Result*
    ret %Result* %x5
}
declare dllimport i8* @__isq__qir__shim__qis__measure__body(i8*, i8*)
define void @__quantum__qis__r__body (i8 %x0, double %x1, %Qubit* %x2) alwaysinline {
entry:
    %x3 = bitcast i8 %x0 to i8
    %x4 = bitcast double %x1 to double
    %x5 = bitcast %Qubit* %x2 to i8*
    call void @__isq__qir__shim__qis__r__body(i8 %x3, double %x4, i8* %x5)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__r__body(i8, double, i8*)
define void @__quantum__qis__r__adj (i8 %x0, double %x1, %Qubit* %x2) alwaysinline {
entry:
    %x3 = bitcast i8 %x0 to i8
    %x4 = bitcast double %x1 to double
    %x5 = bitcast %Qubit* %x2 to i8*
    call void @__isq__qir__shim__qis__r__adj(i8 %x3, double %x4, i8* %x5)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__r__adj(i8, double, i8*)
define void @__quantum__qis__r__ctl (%Array* %x0, { i2, double, %Qubit* }* %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %x1, i64 0, i32 0
    %x4 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %x1, i64 0, i32 1
    %x5 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %x1, i64 0, i32 2
    %x6 = load i2, i2* %x3
    %x9 = load double, double* %x4
    %x7 = load %Qubit*, %Qubit** %x5
    %x8 = zext %Pauli %x6 to i8
    %x10 = bitcast %Qubit* %x7 to i8*
    call void @__isq__qir__shim__qis__r__ctl(i8* %x2, i8 %x8, double %x9, i8* %x10)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__r__ctl(i8*, i8, double, i8*)
define void @__quantum__qis__r__ctladj (%Array* %x0, { i2, double, %Qubit* }* %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %x1, i64 0, i32 0
    %x4 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %x1, i64 0, i32 1
    %x5 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %x1, i64 0, i32 2
    %x6 = load i2, i2* %x3
    %x9 = load double, double* %x4
    %x7 = load %Qubit*, %Qubit** %x5
    %x8 = zext %Pauli %x6 to i8
    %x10 = bitcast %Qubit* %x7 to i8*
    call void @__isq__qir__shim__qis__r__ctladj(i8* %x2, i8 %x8, double %x9, i8* %x10)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__r__ctladj(i8*, i8, double, i8*)
define void @__quantum__qis__s__body (%Qubit* %x0) alwaysinline {
entry:
    %x1 = bitcast %Qubit* %x0 to i8*
    call void @__isq__qir__shim__qis__s__body(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__s__body(i8*)
define void @__quantum__qis__s__adj (%Qubit* %x0) alwaysinline {
entry:
    %x1 = bitcast %Qubit* %x0 to i8*
    call void @__isq__qir__shim__qis__s__adj(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__s__adj(i8*)
define void @__quantum__qis__s__ctl (%Array* %x0, %Qubit* %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = bitcast %Qubit* %x1 to i8*
    call void @__isq__qir__shim__qis__s__ctl(i8* %x2, i8* %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__s__ctl(i8*, i8*)
define void @__quantum__qis__s__ctladj (%Array* %x0, %Qubit* %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = bitcast %Qubit* %x1 to i8*
    call void @__isq__qir__shim__qis__s__ctladj(i8* %x2, i8* %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__s__ctladj(i8*, i8*)
define void @__quantum__qis__t__body (%Qubit* %x0) alwaysinline {
entry:
    %x1 = bitcast %Qubit* %x0 to i8*
    call void @__isq__qir__shim__qis__t__body(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__t__body(i8*)
define void @__quantum__qis__t__adj (%Qubit* %x0) alwaysinline {
entry:
    %x1 = bitcast %Qubit* %x0 to i8*
    call void @__isq__qir__shim__qis__t__adj(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__t__adj(i8*)
define void @__quantum__qis__t__ctl (%Array* %x0, %Qubit* %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = bitcast %Qubit* %x1 to i8*
    call void @__isq__qir__shim__qis__t__ctl(i8* %x2, i8* %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__t__ctl(i8*, i8*)
define void @__quantum__qis__t__ctladj (%Array* %x0, %Qubit* %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = bitcast %Qubit* %x1 to i8*
    call void @__isq__qir__shim__qis__t__ctladj(i8* %x2, i8* %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__t__ctladj(i8*, i8*)
define void @__quantum__qis__x__body (%Qubit* %x0) alwaysinline {
entry:
    %x1 = bitcast %Qubit* %x0 to i8*
    call void @__isq__qir__shim__qis__x__body(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__x__body(i8*)
define void @__quantum__qis__x__ctl (%Array* %x0, %Qubit* %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = bitcast %Qubit* %x1 to i8*
    call void @__isq__qir__shim__qis__x__ctl(i8* %x2, i8* %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__x__ctl(i8*, i8*)
define void @__quantum__qis__y__body (%Qubit* %x0) alwaysinline {
entry:
    %x1 = bitcast %Qubit* %x0 to i8*
    call void @__isq__qir__shim__qis__y__body(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__y__body(i8*)
define void @__quantum__qis__y__ctl (%Array* %x0, %Qubit* %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = bitcast %Qubit* %x1 to i8*
    call void @__isq__qir__shim__qis__y__ctl(i8* %x2, i8* %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__y__ctl(i8*, i8*)
define void @__quantum__qis__z__body (%Qubit* %x0) alwaysinline {
entry:
    %x1 = bitcast %Qubit* %x0 to i8*
    call void @__isq__qir__shim__qis__z__body(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__z__body(i8*)
define void @__quantum__qis__z__ctl (%Array* %x0, %Qubit* %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = bitcast %Qubit* %x1 to i8*
    call void @__isq__qir__shim__qis__z__ctl(i8* %x2, i8* %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__z__ctl(i8*, i8*)
define void @__quantum__qis__dumpmachine__body (i8* %x0) alwaysinline {
entry:
    %x1 = bitcast i8* %x0 to i8*
    call void @__isq__qir__shim__qis__dumpmachine__body(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__dumpmachine__body(i8*)
define void @__quantum__qis__dumpregister__body (i8* %x0, %Array* %x1) alwaysinline {
entry:
    %x2 = bitcast i8* %x0 to i8*
    %x3 = bitcast %Array* %x1 to i8*
    call void @__isq__qir__shim__qis__dumpregister__body(i8* %x2, i8* %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__dumpregister__body(i8*, i8*)
