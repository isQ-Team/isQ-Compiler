%Range = type { i64, i64, i64 }
%Tuple = type opaque
%Qubit = type opaque
%Result = type opaque
%Array = type opaque
%Callable = type opaque
%BigInt = type opaque
%Pauli = type i2
%String = type opaque
define double @__quantum__qis__nan__body () alwaysinline {
entry:
    %x0 = call double @__isq__qir__shim__qis__nan__body()
    %x1 = bitcast double %x0 to double
    ret double %x1
}
declare dllimport double @__isq__qir__shim__qis__nan__body()
define i1 @__quantum__qis__isnan__body (double %x0) alwaysinline {
entry:
    %x1 = bitcast double %x0 to double
    %x2 = call i1 @__isq__qir__shim__qis__isnan__body(double %x1)
    %x3 = bitcast i1 %x2 to i1
    ret i1 %x3
}
declare dllimport i1 @__isq__qir__shim__qis__isnan__body(double)
define double @__quantum__qis__infinity__body () alwaysinline {
entry:
    %x0 = call double @__isq__qir__shim__qis__infinity__body()
    %x1 = bitcast double %x0 to double
    ret double %x1
}
declare dllimport double @__isq__qir__shim__qis__infinity__body()
define i1 @__quantum__qis__isinf__body (double %x0) alwaysinline {
entry:
    %x1 = bitcast double %x0 to double
    %x2 = call i1 @__isq__qir__shim__qis__isinf__body(double %x1)
    %x3 = bitcast i1 %x2 to i1
    ret i1 %x3
}
declare dllimport i1 @__isq__qir__shim__qis__isinf__body(double)
define i1 @__quantum__qis__isnegativeinfinity__body (double %x0) alwaysinline {
entry:
    %x1 = bitcast double %x0 to double
    %x2 = call i1 @__isq__qir__shim__qis__isnegativeinfinity__body(double %x1)
    %x3 = bitcast i1 %x2 to i1
    ret i1 %x3
}
declare dllimport i1 @__isq__qir__shim__qis__isnegativeinfinity__body(double)
define double @__quantum__qis__sin__body (double %x0) alwaysinline {
entry:
    %x1 = bitcast double %x0 to double
    %x2 = call double @__isq__qir__shim__qis__sin__body(double %x1)
    %x3 = bitcast double %x2 to double
    ret double %x3
}
declare dllimport double @__isq__qir__shim__qis__sin__body(double)
define double @__quantum__qis__cos__body (double %x0) alwaysinline {
entry:
    %x1 = bitcast double %x0 to double
    %x2 = call double @__isq__qir__shim__qis__cos__body(double %x1)
    %x3 = bitcast double %x2 to double
    ret double %x3
}
declare dllimport double @__isq__qir__shim__qis__cos__body(double)
define double @__quantum__qis__tan__body (double %x0) alwaysinline {
entry:
    %x1 = bitcast double %x0 to double
    %x2 = call double @__isq__qir__shim__qis__tan__body(double %x1)
    %x3 = bitcast double %x2 to double
    ret double %x3
}
declare dllimport double @__isq__qir__shim__qis__tan__body(double)
define double @__quantum__qis__arctan2__body (double %x0, double %x1) alwaysinline {
entry:
    %x2 = bitcast double %x0 to double
    %x3 = bitcast double %x1 to double
    %x4 = call double @__isq__qir__shim__qis__arctan2__body(double %x2, double %x3)
    %x5 = bitcast double %x4 to double
    ret double %x5
}
declare dllimport double @__isq__qir__shim__qis__arctan2__body(double, double)
define double @__quantum__qis__sinh__body (double %x0) alwaysinline {
entry:
    %x1 = bitcast double %x0 to double
    %x2 = call double @__isq__qir__shim__qis__sinh__body(double %x1)
    %x3 = bitcast double %x2 to double
    ret double %x3
}
declare dllimport double @__isq__qir__shim__qis__sinh__body(double)
define double @__quantum__qis__cosh__body (double %x0) alwaysinline {
entry:
    %x1 = bitcast double %x0 to double
    %x2 = call double @__isq__qir__shim__qis__cosh__body(double %x1)
    %x3 = bitcast double %x2 to double
    ret double %x3
}
declare dllimport double @__isq__qir__shim__qis__cosh__body(double)
define double @__quantum__qis__tanh__body (double %x0) alwaysinline {
entry:
    %x1 = bitcast double %x0 to double
    %x2 = call double @__isq__qir__shim__qis__tanh__body(double %x1)
    %x3 = bitcast double %x2 to double
    ret double %x3
}
declare dllimport double @__isq__qir__shim__qis__tanh__body(double)
define double @__quantum__qis__arcsin__body (double %x0) alwaysinline {
entry:
    %x1 = bitcast double %x0 to double
    %x2 = call double @__isq__qir__shim__qis__arcsin__body(double %x1)
    %x3 = bitcast double %x2 to double
    ret double %x3
}
declare dllimport double @__isq__qir__shim__qis__arcsin__body(double)
define double @__quantum__qis__arccos__body (double %x0) alwaysinline {
entry:
    %x1 = bitcast double %x0 to double
    %x2 = call double @__isq__qir__shim__qis__arccos__body(double %x1)
    %x3 = bitcast double %x2 to double
    ret double %x3
}
declare dllimport double @__isq__qir__shim__qis__arccos__body(double)
define double @__quantum__qis__arctan__body (double %x0) alwaysinline {
entry:
    %x1 = bitcast double %x0 to double
    %x2 = call double @__isq__qir__shim__qis__arctan__body(double %x1)
    %x3 = bitcast double %x2 to double
    ret double %x3
}
declare dllimport double @__isq__qir__shim__qis__arctan__body(double)
define double @__quantum__qis__sqrt__body (double %x0) alwaysinline {
entry:
    %x1 = bitcast double %x0 to double
    %x2 = call double @__isq__qir__shim__qis__sqrt__body(double %x1)
    %x3 = bitcast double %x2 to double
    ret double %x3
}
declare dllimport double @__isq__qir__shim__qis__sqrt__body(double)
define double @__quantum__qis__log__body (double %x0) alwaysinline {
entry:
    %x1 = bitcast double %x0 to double
    %x2 = call double @__isq__qir__shim__qis__log__body(double %x1)
    %x3 = bitcast double %x2 to double
    ret double %x3
}
declare dllimport double @__isq__qir__shim__qis__log__body(double)
define double @__quantum__qis__ieeeremainder__body (double %x0, double %x1) alwaysinline {
entry:
    %x2 = bitcast double %x0 to double
    %x3 = bitcast double %x1 to double
    %x4 = call double @__isq__qir__shim__qis__ieeeremainder__body(double %x2, double %x3)
    %x5 = bitcast double %x4 to double
    ret double %x5
}
declare dllimport double @__isq__qir__shim__qis__ieeeremainder__body(double, double)
define i64 @__quantum__qis__drawrandomint__body (i64 %x0, i64 %x1) alwaysinline {
entry:
    %x2 = bitcast i64 %x0 to i64
    %x3 = bitcast i64 %x1 to i64
    %x4 = call i64 @__isq__qir__shim__qis__drawrandomint__body(i64 %x2, i64 %x3)
    %x5 = bitcast i64 %x4 to i64
    ret i64 %x5
}
declare dllimport i64 @__isq__qir__shim__qis__drawrandomint__body(i64, i64)
define double @__quantum__qis__drawrandomdouble__body (double %x0, double %x1) alwaysinline {
entry:
    %x2 = bitcast double %x0 to double
    %x3 = bitcast double %x1 to double
    %x4 = call double @__isq__qir__shim__qis__drawrandomdouble__body(double %x2, double %x3)
    %x5 = bitcast double %x4 to double
    ret double %x5
}
declare dllimport double @__isq__qir__shim__qis__drawrandomdouble__body(double, double)
<<<<<<< HEAD
=======
define void @__quantum__qis__applyifelseintrinsic__body (%Result* %x0, %Callable* %x1, %Callable* %x2) alwaysinline {
entry:
    %x3 = bitcast %Result* %x0 to i8*
    %x4 = bitcast %Callable* %x1 to i8*
    %x5 = bitcast %Callable* %x2 to i8*
    call void @__isq__qir__shim__qis__applyifelseintrinsic__body(i8* %x3, i8* %x4, i8* %x5)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__applyifelseintrinsic__body(i8*, i8*, i8*)
define void @__quantum__qis__applyconditionallyinstrinsic__body (%Array* %x0, %Array* %x1, %Callable* %x2, %Callable* %x3) alwaysinline {
entry:
    %x4 = bitcast %Array* %x0 to i8*
    %x5 = bitcast %Array* %x1 to i8*
    %x6 = bitcast %Callable* %x2 to i8*
    %x7 = bitcast %Callable* %x3 to i8*
    call void @__isq__qir__shim__qis__applyconditionallyinstrinsic__body(i8* %x4, i8* %x5, i8* %x6, i8* %x7)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__applyconditionallyinstrinsic__body(i8*, i8*, i8*, i8*)
>>>>>>> merge
define void @__quantum__qis__assertmeasurementprobability__body (%Array* %x0, %Array* %x1, %Result* %x2, double %x3, %String* %x4, double %x5) alwaysinline {
entry:
    %x6 = bitcast %Array* %x0 to i8*
    %x7 = bitcast %Array* %x1 to i8*
    %x8 = bitcast %Result* %x2 to i8*
    %x9 = bitcast double %x3 to double
    %x10 = bitcast %String* %x4 to i8*
    %x11 = bitcast double %x5 to double
    call void @__isq__qir__shim__qis__assertmeasurementprobability__body(i8* %x6, i8* %x7, i8* %x8, double %x9, i8* %x10, double %x11)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__assertmeasurementprobability__body(i8*, i8*, i8*, double, i8*, double)
define void @__quantum__qis__assertmeasurementprobability__ctl (%Array* %x0, { %Array, %Array, %Result, double, %String, double }* %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = bitcast { %Array, %Array, %Result, double, %String, double }* %x1 to i8*
    call void @__isq__qir__shim__qis__assertmeasurementprobability__ctl(i8* %x2, i8* %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__qis__assertmeasurementprobability__ctl(i8*, i8*)
