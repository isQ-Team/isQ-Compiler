%Range = type { i64, i64, i64 }
%Tuple = type opaque
%Qubit = type opaque
%Result = type opaque
%Array = type opaque
%Callable = type opaque
%BigInt = type opaque
%Pauli = type i2
%String = type opaque
define %Array* @__quantum__rt__array_concatenate (%Array* %x0, %Array* %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = bitcast %Array* %x1 to i8*
    %x4 = call i8* @__isq__qir__shim__rt__array_concatenate(i8* %x2, i8* %x3)
    %x5 = bitcast i8* %x4 to %Array*
    ret %Array* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__array_concatenate(i8*, i8*)
define %Array* @__quantum__rt__array_copy (%Array* %x0, i1 %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = bitcast i1 %x1 to i1
    %x4 = call i8* @__isq__qir__shim__rt__array_copy(i8* %x2, i1 %x3)
    %x5 = bitcast i8* %x4 to %Array*
    ret %Array* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__array_copy(i8*, i1)
define %Array* @__quantum__rt__array_create (i32 %x0, i32 %x1, i64* %x2) alwaysinline {
entry:
    %x3 = bitcast i32 %x0 to i32
    %x4 = bitcast i32 %x1 to i32
    %x5 = bitcast i64* %x2 to i8*
    %x6 = call i8* @__isq__qir__shim__rt__array_create(i32 %x3, i32 %x4, i8* %x5)
    %x7 = bitcast i8* %x6 to %Array*
    ret %Array* %x7
}
declare dllimport i8* @__isq__qir__shim__rt__array_create(i32, i32, i8*)
define %Array* @__quantum__rt__array_create_1d (i32 %x0, i64 %x1) alwaysinline {
entry:
    %x2 = bitcast i32 %x0 to i32
    %x3 = bitcast i64 %x1 to i64
    %x4 = call i8* @__isq__qir__shim__rt__array_create_1d(i32 %x2, i64 %x3)
    %x5 = bitcast i8* %x4 to %Array*
    ret %Array* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__array_create_1d(i32, i64)
define i32 @__quantum__rt__array_get_dim (%Array* %x0) alwaysinline {
entry:
    %x1 = bitcast %Array* %x0 to i8*
    %x2 = call i32 @__isq__qir__shim__rt__array_get_dim(i8* %x1)
    %x3 = bitcast i32 %x2 to i32
    ret i32 %x3
}
declare dllimport i32 @__isq__qir__shim__rt__array_get_dim(i8*)
define i8* @__quantum__rt__array_get_element_ptr (%Array* %x0, i64* %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = bitcast i64* %x1 to i8*
    %x4 = call i8* @__isq__qir__shim__rt__array_get_element_ptr(i8* %x2, i8* %x3)
    %x5 = bitcast i8* %x4 to i8*
    ret i8* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__array_get_element_ptr(i8*, i8*)
define i8* @__quantum__rt__array_get_element_ptr_1d (%Array* %x0, i64 %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = bitcast i64 %x1 to i64
    %x4 = call i8* @__isq__qir__shim__rt__array_get_element_ptr_1d(i8* %x2, i64 %x3)
    %x5 = bitcast i8* %x4 to i8*
    ret i8* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__array_get_element_ptr_1d(i8*, i64)
define i64 @__quantum__rt__array_get_size (%Array* %x0, i32 %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = bitcast i32 %x1 to i32
    %x4 = call i64 @__isq__qir__shim__rt__array_get_size(i8* %x2, i32 %x3)
    %x5 = bitcast i64 %x4 to i64
    ret i64 %x5
}
declare dllimport i64 @__isq__qir__shim__rt__array_get_size(i8*, i32)
define i64 @__quantum__rt__array_get_size_1d (%Array* %x0) alwaysinline {
entry:
    %x1 = bitcast %Array* %x0 to i8*
    %x2 = call i64 @__isq__qir__shim__rt__array_get_size_1d(i8* %x1)
    %x3 = bitcast i64 %x2 to i64
    ret i64 %x3
}
declare dllimport i64 @__isq__qir__shim__rt__array_get_size_1d(i8*)
define %Array* @__quantum__rt__array_project (%Array* %x0, i32 %x1, i64 %x2, i1 %x3) alwaysinline {
entry:
    %x4 = bitcast %Array* %x0 to i8*
    %x5 = bitcast i32 %x1 to i32
    %x6 = bitcast i64 %x2 to i64
    %x7 = bitcast i1 %x3 to i1
    %x8 = call i8* @__isq__qir__shim__rt__array_project(i8* %x4, i32 %x5, i64 %x6, i1 %x7)
    %x9 = bitcast i8* %x8 to %Array*
    ret %Array* %x9
}
declare dllimport i8* @__isq__qir__shim__rt__array_project(i8*, i32, i64, i1)
define %Array* @__quantum__rt__array_slice (%Array* %x0, i32 %x1, %Range* %x2, i1 %x3) alwaysinline {
entry:
    %x4 = bitcast %Array* %x0 to i8*
    %x5 = bitcast i32 %x1 to i32
    %x6 = bitcast %Range* %x2 to i8*
    %x7 = bitcast i1 %x3 to i1
    %x8 = call i8* @__isq__qir__shim__rt__array_slice(i8* %x4, i32 %x5, i8* %x6, i1 %x7)
    %x9 = bitcast i8* %x8 to %Array*
    ret %Array* %x9
}
declare dllimport i8* @__isq__qir__shim__rt__array_slice(i8*, i32, i8*, i1)
define %Array* @__quantum__rt__array_slice_1d (%Array* %x0, %Range* %x1, i1 %x2) alwaysinline {
entry:
    %x3 = bitcast %Array* %x0 to i8*
    %x4 = bitcast %Range* %x1 to i8*
    %x5 = bitcast i1 %x2 to i1
    %x6 = call i8* @__isq__qir__shim__rt__array_slice_1d(i8* %x3, i8* %x4, i1 %x5)
    %x7 = bitcast i8* %x6 to %Array*
    ret %Array* %x7
}
declare dllimport i8* @__isq__qir__shim__rt__array_slice_1d(i8*, i8*, i1)
define void @__quantum__rt__array_update_alias_count (%Array* %x0, i32 %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = bitcast i32 %x1 to i32
    call void @__isq__qir__shim__rt__array_update_alias_count(i8* %x2, i32 %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__array_update_alias_count(i8*, i32)
define void @__quantum__rt__array_update_reference_count (%Array* %x0, i32 %x1) alwaysinline {
entry:
    %x2 = bitcast %Array* %x0 to i8*
    %x3 = bitcast i32 %x1 to i32
    call void @__isq__qir__shim__rt__array_update_reference_count(i8* %x2, i32 %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__array_update_reference_count(i8*, i32)
define %BigInt* @__quantum__rt__bigint_add (%BigInt* %x0, %BigInt* %x1) alwaysinline {
entry:
    %x2 = bitcast %BigInt* %x0 to i8*
    %x3 = bitcast %BigInt* %x1 to i8*
    %x4 = call i8* @__isq__qir__shim__rt__bigint_add(i8* %x2, i8* %x3)
    %x5 = bitcast i8* %x4 to %BigInt*
    ret %BigInt* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__bigint_add(i8*, i8*)
define %BigInt* @__quantum__rt__bigint_bitand (%BigInt* %x0, %BigInt* %x1) alwaysinline {
entry:
    %x2 = bitcast %BigInt* %x0 to i8*
    %x3 = bitcast %BigInt* %x1 to i8*
    %x4 = call i8* @__isq__qir__shim__rt__bigint_bitand(i8* %x2, i8* %x3)
    %x5 = bitcast i8* %x4 to %BigInt*
    ret %BigInt* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__bigint_bitand(i8*, i8*)
define %BigInt* @__quantum__rt__bigint_bitor (%BigInt* %x0, %BigInt* %x1) alwaysinline {
entry:
    %x2 = bitcast %BigInt* %x0 to i8*
    %x3 = bitcast %BigInt* %x1 to i8*
    %x4 = call i8* @__isq__qir__shim__rt__bigint_bitor(i8* %x2, i8* %x3)
    %x5 = bitcast i8* %x4 to %BigInt*
    ret %BigInt* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__bigint_bitor(i8*, i8*)
define %BigInt* @__quantum__rt__bigint_bitxor (%BigInt* %x0, %BigInt* %x1) alwaysinline {
entry:
    %x2 = bitcast %BigInt* %x0 to i8*
    %x3 = bitcast %BigInt* %x1 to i8*
    %x4 = call i8* @__isq__qir__shim__rt__bigint_bitxor(i8* %x2, i8* %x3)
    %x5 = bitcast i8* %x4 to %BigInt*
    ret %BigInt* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__bigint_bitxor(i8*, i8*)
define %BigInt* @__quantum__rt__bigint_create_array (i32 %x0, i8* %x1) alwaysinline {
entry:
    %x2 = bitcast i32 %x0 to i32
    %x3 = bitcast i8* %x1 to i8*
    %x4 = call i8* @__isq__qir__shim__rt__bigint_create_array(i32 %x2, i8* %x3)
    %x5 = bitcast i8* %x4 to %BigInt*
    ret %BigInt* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__bigint_create_array(i32, i8*)
define %BigInt* @__quantum__rt__bigint_create_i64 (i64 %x0) alwaysinline {
entry:
    %x1 = bitcast i64 %x0 to i64
    %x2 = call i8* @__isq__qir__shim__rt__bigint_create_i64(i64 %x1)
    %x3 = bitcast i8* %x2 to %BigInt*
    ret %BigInt* %x3
}
declare dllimport i8* @__isq__qir__shim__rt__bigint_create_i64(i64)
define %BigInt* @__quantum__rt__bigint_divide (%BigInt* %x0, %BigInt* %x1) alwaysinline {
entry:
    %x2 = bitcast %BigInt* %x0 to i8*
    %x3 = bitcast %BigInt* %x1 to i8*
    %x4 = call i8* @__isq__qir__shim__rt__bigint_divide(i8* %x2, i8* %x3)
    %x5 = bitcast i8* %x4 to %BigInt*
    ret %BigInt* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__bigint_divide(i8*, i8*)
define i1 @__quantum__rt__bigint_equal (%BigInt* %x0, %BigInt* %x1) alwaysinline {
entry:
    %x2 = bitcast %BigInt* %x0 to i8*
    %x3 = bitcast %BigInt* %x1 to i8*
    %x4 = call i1 @__isq__qir__shim__rt__bigint_equal(i8* %x2, i8* %x3)
    %x5 = bitcast i1 %x4 to i1
    ret i1 %x5
}
declare dllimport i1 @__isq__qir__shim__rt__bigint_equal(i8*, i8*)
define i8* @__quantum__rt__bigint_get_data (%BigInt* %x0) alwaysinline {
entry:
    %x1 = bitcast %BigInt* %x0 to i8*
    %x2 = call i8* @__isq__qir__shim__rt__bigint_get_data(i8* %x1)
    %x3 = bitcast i8* %x2 to i8*
    ret i8* %x3
}
declare dllimport i8* @__isq__qir__shim__rt__bigint_get_data(i8*)
define i32 @__quantum__rt__bigint_get_length (%BigInt* %x0) alwaysinline {
entry:
    %x1 = bitcast %BigInt* %x0 to i8*
    %x2 = call i32 @__isq__qir__shim__rt__bigint_get_length(i8* %x1)
    %x3 = bitcast i32 %x2 to i32
    ret i32 %x3
}
declare dllimport i32 @__isq__qir__shim__rt__bigint_get_length(i8*)
define i1 @__quantum__rt__bigint_greater (%BigInt* %x0, %BigInt* %x1) alwaysinline {
entry:
    %x2 = bitcast %BigInt* %x0 to i8*
    %x3 = bitcast %BigInt* %x1 to i8*
    %x4 = call i1 @__isq__qir__shim__rt__bigint_greater(i8* %x2, i8* %x3)
    %x5 = bitcast i1 %x4 to i1
    ret i1 %x5
}
declare dllimport i1 @__isq__qir__shim__rt__bigint_greater(i8*, i8*)
define i1 @__quantum__rt__bigint_greater_eq (%BigInt* %x0, %BigInt* %x1) alwaysinline {
entry:
    %x2 = bitcast %BigInt* %x0 to i8*
    %x3 = bitcast %BigInt* %x1 to i8*
    %x4 = call i1 @__isq__qir__shim__rt__bigint_greater_eq(i8* %x2, i8* %x3)
    %x5 = bitcast i1 %x4 to i1
    ret i1 %x5
}
declare dllimport i1 @__isq__qir__shim__rt__bigint_greater_eq(i8*, i8*)
define %BigInt* @__quantum__rt__bigint_modulus (%BigInt* %x0, %BigInt* %x1) alwaysinline {
entry:
    %x2 = bitcast %BigInt* %x0 to i8*
    %x3 = bitcast %BigInt* %x1 to i8*
    %x4 = call i8* @__isq__qir__shim__rt__bigint_modulus(i8* %x2, i8* %x3)
    %x5 = bitcast i8* %x4 to %BigInt*
    ret %BigInt* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__bigint_modulus(i8*, i8*)
define %BigInt* @__quantum__rt__bigint_multiply (%BigInt* %x0, %BigInt* %x1) alwaysinline {
entry:
    %x2 = bitcast %BigInt* %x0 to i8*
    %x3 = bitcast %BigInt* %x1 to i8*
    %x4 = call i8* @__isq__qir__shim__rt__bigint_multiply(i8* %x2, i8* %x3)
    %x5 = bitcast i8* %x4 to %BigInt*
    ret %BigInt* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__bigint_multiply(i8*, i8*)
define %BigInt* @__quantum__rt__bigint_negate (%BigInt* %x0) alwaysinline {
entry:
    %x1 = bitcast %BigInt* %x0 to i8*
    %x2 = call i8* @__isq__qir__shim__rt__bigint_negate(i8* %x1)
    %x3 = bitcast i8* %x2 to %BigInt*
    ret %BigInt* %x3
}
declare dllimport i8* @__isq__qir__shim__rt__bigint_negate(i8*)
define %BigInt* @__quantum__rt__bigint_power (%BigInt* %x0, i32 %x1) alwaysinline {
entry:
    %x2 = bitcast %BigInt* %x0 to i8*
    %x3 = bitcast i32 %x1 to i32
    %x4 = call i8* @__isq__qir__shim__rt__bigint_power(i8* %x2, i32 %x3)
    %x5 = bitcast i8* %x4 to %BigInt*
    ret %BigInt* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__bigint_power(i8*, i32)
define %BigInt* @__quantum__rt__bigint_shiftleft (%BigInt* %x0, i64 %x1) alwaysinline {
entry:
    %x2 = bitcast %BigInt* %x0 to i8*
    %x3 = bitcast i64 %x1 to i64
    %x4 = call i8* @__isq__qir__shim__rt__bigint_shiftleft(i8* %x2, i64 %x3)
    %x5 = bitcast i8* %x4 to %BigInt*
    ret %BigInt* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__bigint_shiftleft(i8*, i64)
define %BigInt* @__quantum__rt__bigint_shiftright (%BigInt* %x0, i64 %x1) alwaysinline {
entry:
    %x2 = bitcast %BigInt* %x0 to i8*
    %x3 = bitcast i64 %x1 to i64
    %x4 = call i8* @__isq__qir__shim__rt__bigint_shiftright(i8* %x2, i64 %x3)
    %x5 = bitcast i8* %x4 to %BigInt*
    ret %BigInt* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__bigint_shiftright(i8*, i64)
define %BigInt* @__quantum__rt__bigint_subtract (%BigInt* %x0, %BigInt* %x1) alwaysinline {
entry:
    %x2 = bitcast %BigInt* %x0 to i8*
    %x3 = bitcast %BigInt* %x1 to i8*
    %x4 = call i8* @__isq__qir__shim__rt__bigint_subtract(i8* %x2, i8* %x3)
    %x5 = bitcast i8* %x4 to %BigInt*
    ret %BigInt* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__bigint_subtract(i8*, i8*)
define %String* @__quantum__rt__bigint_to_string (%BigInt* %x0) alwaysinline {
entry:
    %x1 = bitcast %BigInt* %x0 to i8*
    %x2 = call i8* @__isq__qir__shim__rt__bigint_to_string(i8* %x1)
    %x3 = bitcast i8* %x2 to %String*
    ret %String* %x3
}
declare dllimport i8* @__isq__qir__shim__rt__bigint_to_string(i8*)
define void @__quantum__rt__bigint_update_reference_count (%BigInt* %x0, i32 %x1) alwaysinline {
entry:
    %x2 = bitcast %BigInt* %x0 to i8*
    %x3 = bitcast i32 %x1 to i32
    call void @__isq__qir__shim__rt__bigint_update_reference_count(i8* %x2, i32 %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__bigint_update_reference_count(i8*, i32)
define %String* @__quantum__rt__bool_to_string (i1 %x0) alwaysinline {
entry:
    %x1 = bitcast i1 %x0 to i1
    %x2 = call i8* @__isq__qir__shim__rt__bool_to_string(i1 %x1)
    %x3 = bitcast i8* %x2 to %String*
    ret %String* %x3
}
declare dllimport i8* @__isq__qir__shim__rt__bool_to_string(i1)
define %Callable* @__quantum__rt__callable_copy (%Callable* %x0, i1 %x1) alwaysinline {
entry:
    %x2 = bitcast %Callable* %x0 to i8*
    %x3 = bitcast i1 %x1 to i1
    %x4 = call i8* @__isq__qir__shim__rt__callable_copy(i8* %x2, i1 %x3)
    %x5 = bitcast i8* %x4 to %Callable*
    ret %Callable* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__callable_copy(i8*, i1)
define %Callable* @__quantum__rt__callable_create ([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* %x0, [2 x void(%Tuple*, i32)*]* %x1, %Tuple* %x2) alwaysinline {
entry:
    %x3 = getelementptr inbounds [4 x void (%Tuple*, %Tuple*, %Tuple*)*],[4 x void (%Tuple*, %Tuple*, %Tuple*)*]* %x0, i32 0, i32 0
    %x7 = load void (%Tuple*, %Tuple*, %Tuple*)*, void (%Tuple*, %Tuple*, %Tuple*)** %x3
    %x11 = bitcast void (%Tuple*, %Tuple*, %Tuple*)* %x7 to i8*
    %x4 = getelementptr inbounds [4 x void (%Tuple*, %Tuple*, %Tuple*)*],[4 x void (%Tuple*, %Tuple*, %Tuple*)*]* %x0, i32 0, i32 1
    %x8 = load void (%Tuple*, %Tuple*, %Tuple*)*, void (%Tuple*, %Tuple*, %Tuple*)** %x4
    %x12 = bitcast void (%Tuple*, %Tuple*, %Tuple*)* %x8 to i8*
    %x5 = getelementptr inbounds [4 x void (%Tuple*, %Tuple*, %Tuple*)*],[4 x void (%Tuple*, %Tuple*, %Tuple*)*]* %x0, i32 0, i32 2
    %x9 = load void (%Tuple*, %Tuple*, %Tuple*)*, void (%Tuple*, %Tuple*, %Tuple*)** %x5
    %x13 = bitcast void (%Tuple*, %Tuple*, %Tuple*)* %x9 to i8*
    %x6 = getelementptr inbounds [4 x void (%Tuple*, %Tuple*, %Tuple*)*],[4 x void (%Tuple*, %Tuple*, %Tuple*)*]* %x0, i32 0, i32 3
    %x10 = load void (%Tuple*, %Tuple*, %Tuple*)*, void (%Tuple*, %Tuple*, %Tuple*)** %x6
    %x14 = bitcast void (%Tuple*, %Tuple*, %Tuple*)* %x10 to i8*
    br label %label_0
label_0:
    %x15 = icmp eq [2 x void(%Tuple*, i32)*]* %x1, null
    br i1 %x15, label %label_2, label %label_1
label_1:
    %x16 = getelementptr inbounds [2 x void(%Tuple*, i32)*], [2 x void(%Tuple*, i32)*]* %x1, i64 0, i64 0
    %x17 = load void(%Tuple*, i32)*, void(%Tuple*, i32)**  %x16
    %x18 = bitcast void(%Tuple*, i32)* %x17 to i8*
    %x19 = getelementptr inbounds [2 x void(%Tuple*, i32)*], [2 x void(%Tuple*, i32)*]* %x1, i64 0, i64 1
    %x20 = load void(%Tuple*, i32)*, void(%Tuple*, i32)** %x19
    %x21 = bitcast void(%Tuple*, i32)* %x20 to i8*
    br label %label_2
label_2:
    %x22 = phi i8* [null, %label_0], [%x18, %label_1]
    %x23 = phi i8* [null, %label_0], [%x21, %label_1]
    %x24 = bitcast %Tuple* %x2 to i8*
    %x25 = call i8* @__isq__qir__shim__rt__callable_create(i8* %x11, i8* %x12, i8* %x13, i8* %x14, i8* %x22, i8* %x23, i8* %x24)
    %x26 = bitcast i8* %x25 to %Callable*
    ret %Callable* %x26
}
declare dllimport i8* @__isq__qir__shim__rt__callable_create(i8*, i8*, i8*, i8*, i8*, i8*, i8*)
define void @__quantum__rt__callable_invoke (%Callable* %x0, %Tuple* %x1, %Tuple* %x2) alwaysinline {
entry:
    %x3 = bitcast %Callable* %x0 to i8*
    %x4 = bitcast %Tuple* %x1 to i8*
    %x5 = bitcast %Tuple* %x2 to i8*
    call void @__isq__qir__shim__rt__callable_invoke(i8* %x3, i8* %x4, i8* %x5)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__callable_invoke(i8*, i8*, i8*)
define void @__quantum__rt__callable_make_adjoint (%Callable* %x0) alwaysinline {
entry:
    %x1 = bitcast %Callable* %x0 to i8*
    call void @__isq__qir__shim__rt__callable_make_adjoint(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__callable_make_adjoint(i8*)
define void @__quantum__rt__callable_make_controlled (%Callable* %x0) alwaysinline {
entry:
    %x1 = bitcast %Callable* %x0 to i8*
    call void @__isq__qir__shim__rt__callable_make_controlled(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__callable_make_controlled(i8*)
define void @__quantum__rt__callable_update_alias_count (%Callable* %x0, i32 %x1) alwaysinline {
entry:
    %x2 = bitcast %Callable* %x0 to i8*
    %x3 = bitcast i32 %x1 to i32
    call void @__isq__qir__shim__rt__callable_update_alias_count(i8* %x2, i32 %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__callable_update_alias_count(i8*, i32)
define void @__quantum__rt__callable_update_reference_count (%Callable* %x0, i32 %x1) alwaysinline {
entry:
    %x2 = bitcast %Callable* %x0 to i8*
    %x3 = bitcast i32 %x1 to i32
    call void @__isq__qir__shim__rt__callable_update_reference_count(i8* %x2, i32 %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__callable_update_reference_count(i8*, i32)
define void @__quantum__rt__capture_update_alias_count (%Callable* %x0, i32 %x1) alwaysinline {
entry:
    %x2 = bitcast %Callable* %x0 to i8*
    %x3 = bitcast i32 %x1 to i32
    call void @__isq__qir__shim__rt__capture_update_alias_count(i8* %x2, i32 %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__capture_update_alias_count(i8*, i32)
define void @__quantum__rt__capture_update_reference_count (%Callable* %x0, i32 %x1) alwaysinline {
entry:
    %x2 = bitcast %Callable* %x0 to i8*
    %x3 = bitcast i32 %x1 to i32
    call void @__isq__qir__shim__rt__capture_update_reference_count(i8* %x2, i32 %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__capture_update_reference_count(i8*, i32)
define %String* @__quantum__rt__double_to_string (double %x0) alwaysinline {
entry:
    %x1 = bitcast double %x0 to double
    %x2 = call i8* @__isq__qir__shim__rt__double_to_string(double %x1)
    %x3 = bitcast i8* %x2 to %String*
    ret %String* %x3
}
declare dllimport i8* @__isq__qir__shim__rt__double_to_string(double)
define void @__quantum__rt__fail (%String* %x0) alwaysinline {
entry:
    %x1 = bitcast %String* %x0 to i8*
    call void @__isq__qir__shim__rt__fail(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__fail(i8*)
define %String* @__quantum__rt__int_to_string (i64 %x0) alwaysinline {
entry:
    %x1 = bitcast i64 %x0 to i64
    %x2 = call i8* @__isq__qir__shim__rt__int_to_string(i64 %x1)
    %x3 = bitcast i8* %x2 to %String*
    ret %String* %x3
}
declare dllimport i8* @__isq__qir__shim__rt__int_to_string(i64)
define void @__quantum__rt__message (%String* %x0) alwaysinline {
entry:
    %x1 = bitcast %String* %x0 to i8*
    call void @__isq__qir__shim__rt__message(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__message(i8*)
define %String* @__quantum__rt__pauli_to_string (%Pauli %x0) alwaysinline {
entry:
    %x1 = zext %Pauli %x0 to i8
    %x2 = call i8* @__isq__qir__shim__rt__pauli_to_string(i8 %x1)
    %x3 = bitcast i8* %x2 to %String*
    ret %String* %x3
}
declare dllimport i8* @__isq__qir__shim__rt__pauli_to_string(i8)
define %Qubit* @__quantum__rt__qubit_allocate () alwaysinline {
entry:
    %x0 = call i8* @__isq__qir__shim__rt__qubit_allocate()
    %x1 = bitcast i8* %x0 to %Qubit*
    ret %Qubit* %x1
}
declare dllimport i8* @__isq__qir__shim__rt__qubit_allocate()
define %Array* @__quantum__rt__qubit_allocate_array (i32 %x0) alwaysinline {
entry:
    %x1 = bitcast i32 %x0 to i32
    %x2 = call i8* @__isq__qir__shim__rt__qubit_allocate_array(i32 %x1)
    %x3 = bitcast i8* %x2 to %Array*
    ret %Array* %x3
}
declare dllimport i8* @__isq__qir__shim__rt__qubit_allocate_array(i32)
define void @__quantum__rt__qubit_release (%Qubit* %x0) alwaysinline {
entry:
    %x1 = bitcast %Qubit* %x0 to i8*
    call void @__isq__qir__shim__rt__qubit_release(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__qubit_release(i8*)
define void @__quantum__rt__qubit_release_array (%Array* %x0) alwaysinline {
entry:
    %x1 = bitcast %Array* %x0 to i8*
    call void @__isq__qir__shim__rt__qubit_release_array(i8* %x1)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__qubit_release_array(i8*)
define %String* @__quantum__rt__qubit_to_string (%Qubit* %x0) alwaysinline {
entry:
    %x1 = bitcast %Qubit* %x0 to i8*
    %x2 = call i8* @__isq__qir__shim__rt__qubit_to_string(i8* %x1)
    %x3 = bitcast i8* %x2 to %String*
    ret %String* %x3
}
declare dllimport i8* @__isq__qir__shim__rt__qubit_to_string(i8*)
define %String* @__quantum__rt__range_to_string (%Range* %x0) alwaysinline {
entry:
    %x1 = bitcast %Range* %x0 to i8*
    %x2 = call i8* @__isq__qir__shim__rt__range_to_string(i8* %x1)
    %x3 = bitcast i8* %x2 to %String*
    ret %String* %x3
}
declare dllimport i8* @__isq__qir__shim__rt__range_to_string(i8*)
define i1 @__quantum__rt__result_equal (%Result* %x0, %Result* %x1) alwaysinline {
entry:
    %x2 = bitcast %Result* %x0 to i8*
    %x3 = bitcast %Result* %x1 to i8*
    %x4 = call i1 @__isq__qir__shim__rt__result_equal(i8* %x2, i8* %x3)
    %x5 = bitcast i1 %x4 to i1
    ret i1 %x5
}
declare dllimport i1 @__isq__qir__shim__rt__result_equal(i8*, i8*)
define %Result* @__quantum__rt__result_get_one () alwaysinline {
entry:
    %x0 = call i8* @__isq__qir__shim__rt__result_get_one()
    %x1 = bitcast i8* %x0 to %Result*
    ret %Result* %x1
}
declare dllimport i8* @__isq__qir__shim__rt__result_get_one()
define %Result* @__quantum__rt__result_get_zero () alwaysinline {
entry:
    %x0 = call i8* @__isq__qir__shim__rt__result_get_zero()
    %x1 = bitcast i8* %x0 to %Result*
    ret %Result* %x1
}
declare dllimport i8* @__isq__qir__shim__rt__result_get_zero()
define %String* @__quantum__rt__result_to_string (%Result* %x0) alwaysinline {
entry:
    %x1 = bitcast %Result* %x0 to i8*
    %x2 = call i8* @__isq__qir__shim__rt__result_to_string(i8* %x1)
    %x3 = bitcast i8* %x2 to %String*
    ret %String* %x3
}
declare dllimport i8* @__isq__qir__shim__rt__result_to_string(i8*)
define void @__quantum__rt__result_update_reference_count (%Result* %x0, i32 %x1) alwaysinline {
entry:
    %x2 = bitcast %Result* %x0 to i8*
    %x3 = bitcast i32 %x1 to i32
    call void @__isq__qir__shim__rt__result_update_reference_count(i8* %x2, i32 %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__result_update_reference_count(i8*, i32)
define %String* @__quantum__rt__string_concatenate (%String* %x0, %String* %x1) alwaysinline {
entry:
    %x2 = bitcast %String* %x0 to i8*
    %x3 = bitcast %String* %x1 to i8*
    %x4 = call i8* @__isq__qir__shim__rt__string_concatenate(i8* %x2, i8* %x3)
    %x5 = bitcast i8* %x4 to %String*
    ret %String* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__string_concatenate(i8*, i8*)
define %String* @__quantum__rt__string_create (i8* %x0) alwaysinline {
entry:
    %x1 = bitcast i8* %x0 to i8*
    %x2 = call i8* @__isq__qir__shim__rt__string_create(i8* %x1)
    %x3 = bitcast i8* %x2 to %String*
    ret %String* %x3
}
declare dllimport i8* @__isq__qir__shim__rt__string_create(i8*)
define i1 @__quantum__rt__string_equal (%String* %x0, %String* %x1) alwaysinline {
entry:
    %x2 = bitcast %String* %x0 to i8*
    %x3 = bitcast %String* %x1 to i8*
    %x4 = call i1 @__isq__qir__shim__rt__string_equal(i8* %x2, i8* %x3)
    %x5 = bitcast i1 %x4 to i1
    ret i1 %x5
}
declare dllimport i1 @__isq__qir__shim__rt__string_equal(i8*, i8*)
define i8* @__quantum__rt__string_get_data (%String* %x0) alwaysinline {
entry:
    %x1 = bitcast %String* %x0 to i8*
    %x2 = call i8* @__isq__qir__shim__rt__string_get_data(i8* %x1)
    %x3 = bitcast i8* %x2 to i8*
    ret i8* %x3
}
declare dllimport i8* @__isq__qir__shim__rt__string_get_data(i8*)
define i32 @__quantum__rt__string_get_length (%String* %x0) alwaysinline {
entry:
    %x1 = bitcast %String* %x0 to i8*
    %x2 = call i32 @__isq__qir__shim__rt__string_get_length(i8* %x1)
    %x3 = bitcast i32 %x2 to i32
    ret i32 %x3
}
declare dllimport i32 @__isq__qir__shim__rt__string_get_length(i8*)
define void @__quantum__rt__string_update_reference_count (%String* %x0, i32 %x1) alwaysinline {
entry:
    %x2 = bitcast %String* %x0 to i8*
    %x3 = bitcast i32 %x1 to i32
    call void @__isq__qir__shim__rt__string_update_reference_count(i8* %x2, i32 %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__string_update_reference_count(i8*, i32)
define %Tuple* @__quantum__rt__tuple_copy (%Tuple* %x0, i1 %x1) alwaysinline {
entry:
    %x2 = bitcast %Tuple* %x0 to i8*
    %x3 = bitcast i1 %x1 to i1
    %x4 = call i8* @__isq__qir__shim__rt__tuple_copy(i8* %x2, i1 %x3)
    %x5 = bitcast i8* %x4 to %Tuple*
    ret %Tuple* %x5
}
declare dllimport i8* @__isq__qir__shim__rt__tuple_copy(i8*, i1)
define %Tuple* @__quantum__rt__tuple_create (i64 %x0) alwaysinline {
entry:
    %x1 = bitcast i64 %x0 to i64
    %x2 = call i8* @__isq__qir__shim__rt__tuple_create(i64 %x1)
    %x3 = bitcast i8* %x2 to %Tuple*
    ret %Tuple* %x3
}
declare dllimport i8* @__isq__qir__shim__rt__tuple_create(i64)
define void @__quantum__rt__tuple_update_alias_count (%Tuple* %x0, i32 %x1) alwaysinline {
entry:
    %x2 = bitcast %Tuple* %x0 to i8*
    %x3 = bitcast i32 %x1 to i32
    call void @__isq__qir__shim__rt__tuple_update_alias_count(i8* %x2, i32 %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__tuple_update_alias_count(i8*, i32)
define void @__quantum__rt__tuple_update_reference_count (%Tuple* %x0, i32 %x1) alwaysinline {
entry:
    %x2 = bitcast %Tuple* %x0 to i8*
    %x3 = bitcast i32 %x1 to i32
    call void @__isq__qir__shim__rt__tuple_update_reference_count(i8* %x2, i32 %x3)
    ret void
}
declare dllimport void @__isq__qir__shim__rt__tuple_update_reference_count(i8*, i32)
