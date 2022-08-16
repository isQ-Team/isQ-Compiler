
%Range = type { i64, i64, i64 }
%Tuple = type opaque
%String = type opaque
%Callable = type opaque
%Array = type opaque
%Result = type opaque

@PauliI = internal constant i2 0
@PauliX = internal constant i2 1
@PauliY = internal constant i2 -1
@PauliZ = internal constant i2 -2
@EmptyRange = internal constant %Range { i64 0, i64 1, i64 -1 }
@0 = internal constant [4 x i8] c"Foo\00"
@1 = internal constant [4 x i8] c"Bar\00"
@Qir__Emission__SillyAssert__FunctionTable = internal constant [4 x void (%Tuple*, %Tuple*, %Tuple*)*] [void (%Tuple*, %Tuple*, %Tuple*)* @Qir__Emission__SillyAssert__body__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Qir__Emission__SillyAssert__adj__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* null, void (%Tuple*, %Tuple*, %Tuple*)* null]
@PartialApplication__1__FunctionTable = internal constant [4 x void (%Tuple*, %Tuple*, %Tuple*)*] [void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__1__body__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__1__adj__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* null, void (%Tuple*, %Tuple*, %Tuple*)* null]
@MemoryManagement__1__FunctionTable = internal constant [2 x void (%Tuple*, i32)*] [void (%Tuple*, i32)* @MemoryManagement__1__RefCount, void (%Tuple*, i32)* @MemoryManagement__1__AliasCount]
@PartialApplication__2__FunctionTable = internal constant [4 x void (%Tuple*, %Tuple*, %Tuple*)*] [void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__2__body__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__2__adj__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* null, void (%Tuple*, %Tuple*, %Tuple*)* null]
@2 = internal constant [24 x i8] c"Finished Callable test!\00"
@3 = internal constant [6 x i8] c"Hello\00"
@4 = internal constant [9 x i8] c", World!\00"
@5 = internal constant [23 x i8] c"Custom Failure Message\00"
@6 = internal constant [38 x i8] c"Original array had incorrect length: \00"
@7 = internal constant [29 x i8] c"Slice had incorrect length: \00"
@8 = internal constant [3 x i8] c", \00"
@9 = internal constant [2 x i8] c"[\00"
@10 = internal constant [2 x i8] c"]\00"
@11 = internal constant [31 x i8] c"Slice had incorrect contents: \00"
@12 = internal constant [54 x i8] c"Slice with every other element had incorrect length: \00"
@13 = internal constant [56 x i8] c"Slice with every other element had incorrect contents: \00"
@14 = internal constant [38 x i8] c"Reversed slice had incorrect length: \00"
@15 = internal constant [40 x i8] c"Reversed slice had incorrect contents: \00"
@16 = internal constant [5 x i8] c" vs \00"
@17 = internal constant [21 x i8] c"Finished Range test!\00"
@18 = internal constant [35 x i8] c"Equivalence of literal One failed!\00"
@19 = internal constant [36 x i8] c"Equivalence of literal Zero failed!\00"
@20 = internal constant [39 x i8] c"Zero and One should not be equivalent!\00"
@21 = internal constant [14 x i8] c"Literal One: \00"
@22 = internal constant [17 x i8] c", Variable One: \00"
@23 = internal constant [15 x i8] c"Literal Zero: \00"
@24 = internal constant [18 x i8] c", Variable Zero: \00"
@25 = internal constant [22 x i8] c"Finished Result test!\00"
@26 = internal constant [28 x i8] c" is not less than or equal \00"
@27 = internal constant [31 x i8] c" is not greater than or equal \00"
@28 = internal constant [7 x i8] c"FooBar\00"
@29 = internal constant [41 x i8] c"Tuple did not preserve expected values: \00"
@30 = internal constant [2 x i8] c"\22\00"
@31 = internal constant [2 x i8] c"!\00"
@32 = internal constant [21 x i8] c"Finished Tuple test!\00"
@33 = internal constant [3 x i8] c"()\00"

define internal void @Qir__Emission__BasicTest__body() {
entry:
  %a = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @0, i32 0, i32 0))
  %b = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @1, i32 0, i32 0))
  %c = call %String* @__quantum__rt__string_concatenate(%String* %a, %String* %b)
  call void @__quantum__rt__message(%String* %c)
  call void @__quantum__rt__string_update_reference_count(%String* %a, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %b, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %c, i32 -1)
  ret void
}

declare %String* @__quantum__rt__string_create(i8*)

declare %String* @__quantum__rt__string_concatenate(%String*, %String*)

declare void @__quantum__rt__message(%String*)

declare void @__quantum__rt__string_update_reference_count(%String*, i32)

define internal void @Qir__Emission__CallableTest__body(i1 %shouldFail) {
entry:
  call void @Qir__Emission__SillyAssert__body(i64 1, i64 1)
  %leq = call %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* @Qir__Emission__SillyAssert__FunctionTable, [2 x void (%Tuple*, i32)*]* null, %Tuple* null)
  call void @__quantum__rt__capture_update_alias_count(%Callable* %leq, i32 1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %leq, i32 1)
  %0 = call %Tuple* @__quantum__rt__tuple_create(i64 mul nuw (i64 ptrtoint (i64* getelementptr (i64, i64* null, i32 1) to i64), i64 2))
  %1 = bitcast %Tuple* %0 to { i64, i64 }*
  %2 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %1, i32 0, i32 0
  %3 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %1, i32 0, i32 1
  store i64 1, i64* %2, align 4
  store i64 2, i64* %3, align 4
  call void @__quantum__rt__callable_invoke(%Callable* %leq, %Tuple* %0, %Tuple* null)
  %geq = call %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* @Qir__Emission__SillyAssert__FunctionTable, [2 x void (%Tuple*, i32)*]* null, %Tuple* null)
  call void @__quantum__rt__callable_make_adjoint(%Callable* %geq)
  call void @__quantum__rt__capture_update_alias_count(%Callable* %geq, i32 1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %geq, i32 1)
  %4 = call %Tuple* @__quantum__rt__tuple_create(i64 mul nuw (i64 ptrtoint (i64* getelementptr (i64, i64* null, i32 1) to i64), i64 2))
  %5 = bitcast %Tuple* %4 to { i64, i64 }*
  %6 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %5, i32 0, i32 0
  %7 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %5, i32 0, i32 1
  store i64 2, i64* %6, align 4
  store i64 1, i64* %7, align 4
  call void @__quantum__rt__callable_invoke(%Callable* %geq, %Tuple* %4, %Tuple* null)
  %8 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Callable*, i64 }* getelementptr ({ %Callable*, i64 }, { %Callable*, i64 }* null, i32 1) to i64))
  %9 = bitcast %Tuple* %8 to { %Callable*, i64 }*
  %10 = getelementptr inbounds { %Callable*, i64 }, { %Callable*, i64 }* %9, i32 0, i32 0
  %11 = getelementptr inbounds { %Callable*, i64 }, { %Callable*, i64 }* %9, i32 0, i32 1
  %12 = call %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* @Qir__Emission__SillyAssert__FunctionTable, [2 x void (%Tuple*, i32)*]* null, %Tuple* null)
  store %Callable* %12, %Callable** %10, align 8
  store i64 0, i64* %11, align 4
  %assertNotNegative = call %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* @PartialApplication__1__FunctionTable, [2 x void (%Tuple*, i32)*]* @MemoryManagement__1__FunctionTable, %Tuple* %8)
  call void @__quantum__rt__capture_update_alias_count(%Callable* %assertNotNegative, i32 1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %assertNotNegative, i32 1)
  %13 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint (i64* getelementptr (i64, i64* null, i32 1) to i64))
  %14 = bitcast %Tuple* %13 to { i64 }*
  %15 = getelementptr inbounds { i64 }, { i64 }* %14, i32 0, i32 0
  store i64 3, i64* %15, align 4
  call void @__quantum__rt__callable_invoke(%Callable* %assertNotNegative, %Tuple* %13, %Tuple* null)
  %16 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Callable*, i64 }* getelementptr ({ %Callable*, i64 }, { %Callable*, i64 }* null, i32 1) to i64))
  %17 = bitcast %Tuple* %16 to { %Callable*, i64 }*
  %18 = getelementptr inbounds { %Callable*, i64 }, { %Callable*, i64 }* %17, i32 0, i32 0
  %19 = getelementptr inbounds { %Callable*, i64 }, { %Callable*, i64 }* %17, i32 0, i32 1
  %20 = call %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* @Qir__Emission__SillyAssert__FunctionTable, [2 x void (%Tuple*, i32)*]* null, %Tuple* null)
  call void @__quantum__rt__callable_make_adjoint(%Callable* %20)
  store %Callable* %20, %Callable** %18, align 8
  store i64 42, i64* %19, align 4
  %geq42 = call %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* @PartialApplication__2__FunctionTable, [2 x void (%Tuple*, i32)*]* @MemoryManagement__1__FunctionTable, %Tuple* %16)
  call void @__quantum__rt__capture_update_alias_count(%Callable* %geq42, i32 1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %geq42, i32 1)
  %21 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint (i64* getelementptr (i64, i64* null, i32 1) to i64))
  %22 = bitcast %Tuple* %21 to { i64 }*
  %23 = getelementptr inbounds { i64 }, { i64 }* %22, i32 0, i32 0
  store i64 43, i64* %23, align 4
  call void @__quantum__rt__callable_invoke(%Callable* %geq42, %Tuple* %21, %Tuple* null)
  br i1 %shouldFail, label %then0__1, label %else__1

then0__1:                                         ; preds = %entry
  %24 = call %Callable* @__quantum__rt__callable_copy(%Callable* %geq42, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %24, i32 1)
  call void @__quantum__rt__callable_make_adjoint(%Callable* %24)
  %25 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint (i64* getelementptr (i64, i64* null, i32 1) to i64))
  %26 = bitcast %Tuple* %25 to { i64 }*
  %27 = getelementptr inbounds { i64 }, { i64 }* %26, i32 0, i32 0
  store i64 43, i64* %27, align 4
  call void @__quantum__rt__callable_invoke(%Callable* %24, %Tuple* %25, %Tuple* null)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %24, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %24, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %25, i32 -1)
  br label %continue__1

else__1:                                          ; preds = %entry
  %28 = call %Callable* @__quantum__rt__callable_copy(%Callable* %geq42, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %28, i32 1)
  call void @__quantum__rt__callable_make_adjoint(%Callable* %28)
  %29 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint (i64* getelementptr (i64, i64* null, i32 1) to i64))
  %30 = bitcast %Tuple* %29 to { i64 }*
  %31 = getelementptr inbounds { i64 }, { i64 }* %30, i32 0, i32 0
  store i64 41, i64* %31, align 4
  call void @__quantum__rt__callable_invoke(%Callable* %28, %Tuple* %29, %Tuple* null)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %28, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %28, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %29, i32 -1)
  br label %continue__1

continue__1:                                      ; preds = %else__1, %then0__1
  %32 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @2, i32 0, i32 0))
  call void @__quantum__rt__message(%String* %32)
  call void @__quantum__rt__capture_update_alias_count(%Callable* %leq, i32 -1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %leq, i32 -1)
  call void @__quantum__rt__capture_update_alias_count(%Callable* %geq, i32 -1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %geq, i32 -1)
  call void @__quantum__rt__capture_update_alias_count(%Callable* %assertNotNegative, i32 -1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %assertNotNegative, i32 -1)
  call void @__quantum__rt__capture_update_alias_count(%Callable* %geq42, i32 -1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %geq42, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %leq, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %leq, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %0, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %geq, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %geq, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %4, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %assertNotNegative, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %assertNotNegative, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %13, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %geq42, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %geq42, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %21, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %32, i32 -1)
  ret void
}

define internal void @Qir__Emission__SillyAssert__body(i64 %a, i64 %b) {
entry:
  %0 = icmp sgt i64 %a, %b
  br i1 %0, label %then0__1, label %continue__1

then0__1:                                         ; preds = %entry
  %1 = call %String* @__quantum__rt__int_to_string(i64 %a)
  %2 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([28 x i8], [28 x i8]* @26, i32 0, i32 0))
  %3 = call %String* @__quantum__rt__string_concatenate(%String* %1, %String* %2)
  call void @__quantum__rt__string_update_reference_count(%String* %1, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %2, i32 -1)
  %4 = call %String* @__quantum__rt__int_to_string(i64 %b)
  %5 = call %String* @__quantum__rt__string_concatenate(%String* %3, %String* %4)
  call void @__quantum__rt__string_update_reference_count(%String* %3, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %4, i32 -1)
  call void @__quantum__rt__fail(%String* %5)
  unreachable

continue__1:                                      ; preds = %entry
  ret void
}

define internal void @Qir__Emission__SillyAssert__body__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { i64, i64 }*
  %1 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %0, i32 0, i32 0
  %2 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %0, i32 0, i32 1
  %3 = load i64, i64* %1, align 4
  %4 = load i64, i64* %2, align 4
  call void @Qir__Emission__SillyAssert__body(i64 %3, i64 %4)
  ret void
}

define internal void @Qir__Emission__SillyAssert__adj__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { i64, i64 }*
  %1 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %0, i32 0, i32 0
  %2 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %0, i32 0, i32 1
  %3 = load i64, i64* %1, align 4
  %4 = load i64, i64* %2, align 4
  call void @Qir__Emission__SillyAssert__adj(i64 %3, i64 %4)
  ret void
}

declare %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]*, [2 x void (%Tuple*, i32)*]*, %Tuple*)

declare void @__quantum__rt__capture_update_alias_count(%Callable*, i32)

declare void @__quantum__rt__callable_update_alias_count(%Callable*, i32)

declare void @__quantum__rt__callable_invoke(%Callable*, %Tuple*, %Tuple*)

declare %Tuple* @__quantum__rt__tuple_create(i64)

declare void @__quantum__rt__callable_make_adjoint(%Callable*)

define internal void @Lifted__PartialApplication__1__body__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %capture-tuple to { %Callable*, i64 }*
  %1 = getelementptr inbounds { %Callable*, i64 }, { %Callable*, i64 }* %0, i32 0, i32 1
  %2 = load i64, i64* %1, align 4
  %3 = bitcast %Tuple* %arg-tuple to { i64 }*
  %4 = getelementptr inbounds { i64 }, { i64 }* %3, i32 0, i32 0
  %5 = load i64, i64* %4, align 4
  %6 = call %Tuple* @__quantum__rt__tuple_create(i64 mul nuw (i64 ptrtoint (i64* getelementptr (i64, i64* null, i32 1) to i64), i64 2))
  %7 = bitcast %Tuple* %6 to { i64, i64 }*
  %8 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %7, i32 0, i32 0
  %9 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %7, i32 0, i32 1
  store i64 %2, i64* %8, align 4
  store i64 %5, i64* %9, align 4
  %10 = getelementptr inbounds { %Callable*, i64 }, { %Callable*, i64 }* %0, i32 0, i32 0
  %11 = load %Callable*, %Callable** %10, align 8
  call void @__quantum__rt__callable_invoke(%Callable* %11, %Tuple* %6, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %6, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__1__adj__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %capture-tuple to { %Callable*, i64 }*
  %1 = getelementptr inbounds { %Callable*, i64 }, { %Callable*, i64 }* %0, i32 0, i32 1
  %2 = load i64, i64* %1, align 4
  %3 = bitcast %Tuple* %arg-tuple to { i64 }*
  %4 = getelementptr inbounds { i64 }, { i64 }* %3, i32 0, i32 0
  %5 = load i64, i64* %4, align 4
  %6 = call %Tuple* @__quantum__rt__tuple_create(i64 mul nuw (i64 ptrtoint (i64* getelementptr (i64, i64* null, i32 1) to i64), i64 2))
  %7 = bitcast %Tuple* %6 to { i64, i64 }*
  %8 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %7, i32 0, i32 0
  %9 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %7, i32 0, i32 1
  store i64 %2, i64* %8, align 4
  store i64 %5, i64* %9, align 4
  %10 = getelementptr inbounds { %Callable*, i64 }, { %Callable*, i64 }* %0, i32 0, i32 0
  %11 = load %Callable*, %Callable** %10, align 8
  %12 = call %Callable* @__quantum__rt__callable_copy(%Callable* %11, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %12, i32 1)
  call void @__quantum__rt__callable_make_adjoint(%Callable* %12)
  call void @__quantum__rt__callable_invoke(%Callable* %12, %Tuple* %6, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %6, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %12, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %12, i32 -1)
  ret void
}

define internal void @MemoryManagement__1__RefCount(%Tuple* %capture-tuple, i32 %count-change) {
entry:
  %0 = bitcast %Tuple* %capture-tuple to { %Callable*, i64 }*
  %1 = getelementptr inbounds { %Callable*, i64 }, { %Callable*, i64 }* %0, i32 0, i32 0
  %2 = load %Callable*, %Callable** %1, align 8
  call void @__quantum__rt__capture_update_reference_count(%Callable* %2, i32 %count-change)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %2, i32 %count-change)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %capture-tuple, i32 %count-change)
  ret void
}

define internal void @MemoryManagement__1__AliasCount(%Tuple* %capture-tuple, i32 %count-change) {
entry:
  %0 = bitcast %Tuple* %capture-tuple to { %Callable*, i64 }*
  %1 = getelementptr inbounds { %Callable*, i64 }, { %Callable*, i64 }* %0, i32 0, i32 0
  %2 = load %Callable*, %Callable** %1, align 8
  call void @__quantum__rt__capture_update_alias_count(%Callable* %2, i32 %count-change)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %2, i32 %count-change)
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %capture-tuple, i32 %count-change)
  ret void
}

define internal void @Lifted__PartialApplication__2__body__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { i64 }*
  %1 = getelementptr inbounds { i64 }, { i64 }* %0, i32 0, i32 0
  %2 = load i64, i64* %1, align 4
  %3 = bitcast %Tuple* %capture-tuple to { %Callable*, i64 }*
  %4 = getelementptr inbounds { %Callable*, i64 }, { %Callable*, i64 }* %3, i32 0, i32 1
  %5 = load i64, i64* %4, align 4
  %6 = call %Tuple* @__quantum__rt__tuple_create(i64 mul nuw (i64 ptrtoint (i64* getelementptr (i64, i64* null, i32 1) to i64), i64 2))
  %7 = bitcast %Tuple* %6 to { i64, i64 }*
  %8 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %7, i32 0, i32 0
  %9 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %7, i32 0, i32 1
  store i64 %2, i64* %8, align 4
  store i64 %5, i64* %9, align 4
  %10 = getelementptr inbounds { %Callable*, i64 }, { %Callable*, i64 }* %3, i32 0, i32 0
  %11 = load %Callable*, %Callable** %10, align 8
  call void @__quantum__rt__callable_invoke(%Callable* %11, %Tuple* %6, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %6, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__2__adj__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { i64 }*
  %1 = getelementptr inbounds { i64 }, { i64 }* %0, i32 0, i32 0
  %2 = load i64, i64* %1, align 4
  %3 = bitcast %Tuple* %capture-tuple to { %Callable*, i64 }*
  %4 = getelementptr inbounds { %Callable*, i64 }, { %Callable*, i64 }* %3, i32 0, i32 1
  %5 = load i64, i64* %4, align 4
  %6 = call %Tuple* @__quantum__rt__tuple_create(i64 mul nuw (i64 ptrtoint (i64* getelementptr (i64, i64* null, i32 1) to i64), i64 2))
  %7 = bitcast %Tuple* %6 to { i64, i64 }*
  %8 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %7, i32 0, i32 0
  %9 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %7, i32 0, i32 1
  store i64 %2, i64* %8, align 4
  store i64 %5, i64* %9, align 4
  %10 = getelementptr inbounds { %Callable*, i64 }, { %Callable*, i64 }* %3, i32 0, i32 0
  %11 = load %Callable*, %Callable** %10, align 8
  %12 = call %Callable* @__quantum__rt__callable_copy(%Callable* %11, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %12, i32 1)
  call void @__quantum__rt__callable_make_adjoint(%Callable* %12)
  call void @__quantum__rt__callable_invoke(%Callable* %12, %Tuple* %6, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %6, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %12, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %12, i32 -1)
  ret void
}

declare %Callable* @__quantum__rt__callable_copy(%Callable*, i1)

declare void @__quantum__rt__capture_update_reference_count(%Callable*, i32)

declare void @__quantum__rt__callable_update_reference_count(%Callable*, i32)

declare void @__quantum__rt__tuple_update_reference_count(%Tuple*, i32)

define internal void @Qir__Emission__SillyAssert__adj(i64 %a, i64 %b) {
entry:
  %0 = icmp slt i64 %a, %b
  br i1 %0, label %then0__1, label %continue__1

then0__1:                                         ; preds = %entry
  %1 = call %String* @__quantum__rt__int_to_string(i64 %a)
  %2 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([31 x i8], [31 x i8]* @27, i32 0, i32 0))
  %3 = call %String* @__quantum__rt__string_concatenate(%String* %1, %String* %2)
  call void @__quantum__rt__string_update_reference_count(%String* %1, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %2, i32 -1)
  %4 = call %String* @__quantum__rt__int_to_string(i64 %b)
  %5 = call %String* @__quantum__rt__string_concatenate(%String* %3, %String* %4)
  call void @__quantum__rt__string_update_reference_count(%String* %3, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %4, i32 -1)
  call void @__quantum__rt__fail(%String* %5)
  unreachable

continue__1:                                      ; preds = %entry
  ret void
}

declare void @__quantum__rt__tuple_update_alias_count(%Tuple*, i32)

define internal %String* @Qir__Emission__ConcatTest__body() {
entry:
  %a = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @3, i32 0, i32 0))
  %b = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @4, i32 0, i32 0))
  %0 = call %String* @__quantum__rt__string_concatenate(%String* %a, %String* %b)
  call void @__quantum__rt__string_update_reference_count(%String* %a, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %b, i32 -1)
  ret %String* %0
}

define internal void @Qir__Emission__CustomFail__body() {
entry:
  %0 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @5, i32 0, i32 0))
  call void @__quantum__rt__fail(%String* %0)
  unreachable
}

declare void @__quantum__rt__fail(%String*)

define internal void @Qir__Emission__RangeTest__body() {
entry:
  %array = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 6)
  %0 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %array, i64 0)
  %1 = bitcast i8* %0 to i64*
  %2 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %array, i64 1)
  %3 = bitcast i8* %2 to i64*
  %4 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %array, i64 2)
  %5 = bitcast i8* %4 to i64*
  %6 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %array, i64 3)
  %7 = bitcast i8* %6 to i64*
  %8 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %array, i64 4)
  %9 = bitcast i8* %8 to i64*
  %10 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %array, i64 5)
  %11 = bitcast i8* %10 to i64*
  store i64 0, i64* %1, align 4
  store i64 1, i64* %3, align 4
  store i64 2, i64* %5, align 4
  store i64 3, i64* %7, align 4
  store i64 4, i64* %9, align 4
  store i64 5, i64* %11, align 4
  call void @__quantum__rt__array_update_alias_count(%Array* %array, i32 1)
  %12 = load %Range, %Range* @EmptyRange, align 4
  %13 = insertvalue %Range %12, i64 0, 0
  %14 = insertvalue %Range %13, i64 1, 1
  %15 = insertvalue %Range %14, i64 2, 2
  %slice = call %Array* @__quantum__rt__array_slice_1d(%Array* %array, %Range %15, i1 true)
  call void @__quantum__rt__array_update_alias_count(%Array* %slice, i32 1)
  %16 = load %Range, %Range* @EmptyRange, align 4
  %17 = insertvalue %Range %16, i64 0, 0
  %18 = insertvalue %Range %17, i64 2, 1
  %19 = insertvalue %Range %18, i64 5, 2
  %everyOther = call %Array* @__quantum__rt__array_slice_1d(%Array* %array, %Range %19, i1 true)
  call void @__quantum__rt__array_update_alias_count(%Array* %everyOther, i32 1)
  %20 = load %Range, %Range* @EmptyRange, align 4
  %21 = insertvalue %Range %20, i64 5, 0
  %22 = insertvalue %Range %21, i64 -1, 1
  %23 = insertvalue %Range %22, i64 0, 2
  %backwards = call %Array* @__quantum__rt__array_slice_1d(%Array* %array, %Range %23, i1 true)
  call void @__quantum__rt__array_update_alias_count(%Array* %backwards, i32 1)
  br i1 false, label %then0__1, label %continue__1

then0__1:                                         ; preds = %entry
  %24 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([38 x i8], [38 x i8]* @6, i32 0, i32 0))
  %25 = call %String* @__quantum__rt__int_to_string(i64 6)
  %26 = call %String* @__quantum__rt__string_concatenate(%String* %24, %String* %25)
  call void @__quantum__rt__string_update_reference_count(%String* %24, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %25, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %slice, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %everyOther, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %backwards, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %slice, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %everyOther, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %backwards, i32 -1)
  call void @__quantum__rt__fail(%String* %26)
  unreachable

continue__1:                                      ; preds = %entry
  %27 = call i64 @__quantum__rt__array_get_size_1d(%Array* %slice)
  %28 = icmp ne i64 %27, 3
  br i1 %28, label %then0__2, label %continue__2

then0__2:                                         ; preds = %continue__1
  %29 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @7, i32 0, i32 0))
  %30 = call %String* @__quantum__rt__int_to_string(i64 %27)
  %31 = call %String* @__quantum__rt__string_concatenate(%String* %29, %String* %30)
  call void @__quantum__rt__string_update_reference_count(%String* %29, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %30, i32 -1)
  %32 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @8, i32 0, i32 0))
  %33 = call %String* @__quantum__rt__string_concatenate(%String* %31, %String* %32)
  call void @__quantum__rt__string_update_reference_count(%String* %31, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %32, i32 -1)
  %34 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @8, i32 0, i32 0))
  %35 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @9, i32 0, i32 0))
  call void @__quantum__rt__string_update_reference_count(%String* %35, i32 1)
  %36 = sub i64 %27, 1
  br label %header__1

continue__2:                                      ; preds = %continue__1
  %37 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %slice, i64 0)
  %38 = bitcast i8* %37 to i64*
  %39 = load i64, i64* %38, align 4
  %40 = icmp ne i64 %39, 0
  %41 = xor i1 %40, true
  br i1 %41, label %condTrue__2, label %condContinue__2

condTrue__2:                                      ; preds = %continue__2
  %42 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %slice, i64 1)
  %43 = bitcast i8* %42 to i64*
  %44 = load i64, i64* %43, align 4
  %45 = icmp ne i64 %44, 1
  br label %condContinue__2

condContinue__2:                                  ; preds = %condTrue__2, %continue__2
  %46 = phi i1 [ %45, %condTrue__2 ], [ %40, %continue__2 ]
  %47 = xor i1 %46, true
  br i1 %47, label %condTrue__3, label %condContinue__3

condTrue__3:                                      ; preds = %condContinue__2
  %48 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %slice, i64 2)
  %49 = bitcast i8* %48 to i64*
  %50 = load i64, i64* %49, align 4
  %51 = icmp ne i64 %50, 2
  br label %condContinue__3

condContinue__3:                                  ; preds = %condTrue__3, %condContinue__2
  %52 = phi i1 [ %51, %condTrue__3 ], [ %46, %condContinue__2 ]
  br i1 %52, label %then0__3, label %continue__3

then0__3:                                         ; preds = %condContinue__3
  %53 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([31 x i8], [31 x i8]* @11, i32 0, i32 0))
  %54 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @8, i32 0, i32 0))
  %55 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @9, i32 0, i32 0))
  call void @__quantum__rt__string_update_reference_count(%String* %55, i32 1)
  %56 = call i64 @__quantum__rt__array_get_size_1d(%Array* %slice)
  %57 = sub i64 %56, 1
  br label %header__2

continue__3:                                      ; preds = %condContinue__3
  %58 = call i64 @__quantum__rt__array_get_size_1d(%Array* %everyOther)
  %59 = icmp ne i64 %58, 3
  br i1 %59, label %then0__4, label %continue__4

then0__4:                                         ; preds = %continue__3
  %60 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([54 x i8], [54 x i8]* @12, i32 0, i32 0))
  %61 = call %String* @__quantum__rt__int_to_string(i64 %58)
  %62 = call %String* @__quantum__rt__string_concatenate(%String* %60, %String* %61)
  call void @__quantum__rt__string_update_reference_count(%String* %60, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %61, i32 -1)
  %63 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @8, i32 0, i32 0))
  %64 = call %String* @__quantum__rt__string_concatenate(%String* %62, %String* %63)
  call void @__quantum__rt__string_update_reference_count(%String* %62, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %63, i32 -1)
  %65 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @8, i32 0, i32 0))
  %66 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @9, i32 0, i32 0))
  call void @__quantum__rt__string_update_reference_count(%String* %66, i32 1)
  %67 = sub i64 %58, 1
  br label %header__3

continue__4:                                      ; preds = %continue__3
  %68 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %everyOther, i64 0)
  %69 = bitcast i8* %68 to i64*
  %70 = load i64, i64* %69, align 4
  %71 = icmp ne i64 %70, 0
  %72 = xor i1 %71, true
  br i1 %72, label %condTrue__6, label %condContinue__6

condTrue__6:                                      ; preds = %continue__4
  %73 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %everyOther, i64 1)
  %74 = bitcast i8* %73 to i64*
  %75 = load i64, i64* %74, align 4
  %76 = icmp ne i64 %75, 2
  br label %condContinue__6

condContinue__6:                                  ; preds = %condTrue__6, %continue__4
  %77 = phi i1 [ %76, %condTrue__6 ], [ %71, %continue__4 ]
  %78 = xor i1 %77, true
  br i1 %78, label %condTrue__7, label %condContinue__7

condTrue__7:                                      ; preds = %condContinue__6
  %79 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %everyOther, i64 2)
  %80 = bitcast i8* %79 to i64*
  %81 = load i64, i64* %80, align 4
  %82 = icmp ne i64 %81, 4
  br label %condContinue__7

condContinue__7:                                  ; preds = %condTrue__7, %condContinue__6
  %83 = phi i1 [ %82, %condTrue__7 ], [ %77, %condContinue__6 ]
  br i1 %83, label %then0__5, label %continue__5

then0__5:                                         ; preds = %condContinue__7
  %84 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([56 x i8], [56 x i8]* @13, i32 0, i32 0))
  %85 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @8, i32 0, i32 0))
  %86 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @9, i32 0, i32 0))
  call void @__quantum__rt__string_update_reference_count(%String* %86, i32 1)
  %87 = call i64 @__quantum__rt__array_get_size_1d(%Array* %everyOther)
  %88 = sub i64 %87, 1
  br label %header__4

continue__5:                                      ; preds = %condContinue__7
  %89 = call i64 @__quantum__rt__array_get_size_1d(%Array* %backwards)
  %90 = icmp ne i64 %89, 6
  br i1 %90, label %then0__6, label %continue__6

then0__6:                                         ; preds = %continue__5
  %91 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([38 x i8], [38 x i8]* @14, i32 0, i32 0))
  %92 = call %String* @__quantum__rt__int_to_string(i64 %89)
  %93 = call %String* @__quantum__rt__string_concatenate(%String* %91, %String* %92)
  call void @__quantum__rt__string_update_reference_count(%String* %91, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %92, i32 -1)
  %94 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @8, i32 0, i32 0))
  %95 = call %String* @__quantum__rt__string_concatenate(%String* %93, %String* %94)
  call void @__quantum__rt__string_update_reference_count(%String* %93, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %94, i32 -1)
  %96 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @8, i32 0, i32 0))
  %97 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @9, i32 0, i32 0))
  call void @__quantum__rt__string_update_reference_count(%String* %97, i32 1)
  %98 = sub i64 %89, 1
  br label %header__5

continue__6:                                      ; preds = %continue__5
  %99 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %backwards, i64 0)
  %100 = bitcast i8* %99 to i64*
  %101 = load i64, i64* %100, align 4
  %102 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %array, i64 5)
  %103 = bitcast i8* %102 to i64*
  %104 = load i64, i64* %103, align 4
  %105 = icmp ne i64 %101, %104
  %106 = xor i1 %105, true
  br i1 %106, label %condTrue__10, label %condContinue__10

condTrue__10:                                     ; preds = %continue__6
  %107 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %backwards, i64 1)
  %108 = bitcast i8* %107 to i64*
  %109 = load i64, i64* %108, align 4
  %110 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %array, i64 4)
  %111 = bitcast i8* %110 to i64*
  %112 = load i64, i64* %111, align 4
  %113 = icmp ne i64 %109, %112
  br label %condContinue__10

condContinue__10:                                 ; preds = %condTrue__10, %continue__6
  %114 = phi i1 [ %113, %condTrue__10 ], [ %105, %continue__6 ]
  %115 = xor i1 %114, true
  br i1 %115, label %condTrue__11, label %condContinue__11

condTrue__11:                                     ; preds = %condContinue__10
  %116 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %backwards, i64 2)
  %117 = bitcast i8* %116 to i64*
  %118 = load i64, i64* %117, align 4
  %119 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %array, i64 3)
  %120 = bitcast i8* %119 to i64*
  %121 = load i64, i64* %120, align 4
  %122 = icmp ne i64 %118, %121
  br label %condContinue__11

condContinue__11:                                 ; preds = %condTrue__11, %condContinue__10
  %123 = phi i1 [ %122, %condTrue__11 ], [ %114, %condContinue__10 ]
  %124 = xor i1 %123, true
  br i1 %124, label %condTrue__12, label %condContinue__12

condTrue__12:                                     ; preds = %condContinue__11
  %125 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %backwards, i64 3)
  %126 = bitcast i8* %125 to i64*
  %127 = load i64, i64* %126, align 4
  %128 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %array, i64 2)
  %129 = bitcast i8* %128 to i64*
  %130 = load i64, i64* %129, align 4
  %131 = icmp ne i64 %127, %130
  br label %condContinue__12

condContinue__12:                                 ; preds = %condTrue__12, %condContinue__11
  %132 = phi i1 [ %131, %condTrue__12 ], [ %123, %condContinue__11 ]
  %133 = xor i1 %132, true
  br i1 %133, label %condTrue__13, label %condContinue__13

condTrue__13:                                     ; preds = %condContinue__12
  %134 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %backwards, i64 4)
  %135 = bitcast i8* %134 to i64*
  %136 = load i64, i64* %135, align 4
  %137 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %array, i64 1)
  %138 = bitcast i8* %137 to i64*
  %139 = load i64, i64* %138, align 4
  %140 = icmp ne i64 %136, %139
  br label %condContinue__13

condContinue__13:                                 ; preds = %condTrue__13, %condContinue__12
  %141 = phi i1 [ %140, %condTrue__13 ], [ %132, %condContinue__12 ]
  %142 = xor i1 %141, true
  br i1 %142, label %condTrue__14, label %condContinue__14

condTrue__14:                                     ; preds = %condContinue__13
  %143 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %backwards, i64 5)
  %144 = bitcast i8* %143 to i64*
  %145 = load i64, i64* %144, align 4
  %146 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %array, i64 0)
  %147 = bitcast i8* %146 to i64*
  %148 = load i64, i64* %147, align 4
  %149 = icmp ne i64 %145, %148
  br label %condContinue__14

condContinue__14:                                 ; preds = %condTrue__14, %condContinue__13
  %150 = phi i1 [ %149, %condTrue__14 ], [ %141, %condContinue__13 ]
  br i1 %150, label %then0__7, label %continue__7

then0__7:                                         ; preds = %condContinue__14
  %151 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @15, i32 0, i32 0))
  %152 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @8, i32 0, i32 0))
  %153 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @9, i32 0, i32 0))
  call void @__quantum__rt__string_update_reference_count(%String* %153, i32 1)
  %154 = call i64 @__quantum__rt__array_get_size_1d(%Array* %backwards)
  %155 = sub i64 %154, 1
  br label %header__6

continue__7:                                      ; preds = %condContinue__14
  %156 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @17, i32 0, i32 0))
  call void @__quantum__rt__message(%String* %156)
  call void @__quantum__rt__array_update_alias_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %slice, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %everyOther, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %backwards, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %slice, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %everyOther, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %backwards, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %156, i32 -1)
  ret void

header__1:                                        ; preds = %exiting__1, %then0__2
  %157 = phi %String* [ %35, %then0__2 ], [ %167, %exiting__1 ]
  %158 = phi i64 [ 0, %then0__2 ], [ %168, %exiting__1 ]
  %159 = icmp sle i64 %158, %36
  br i1 %159, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %160 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %slice, i64 %158)
  %161 = bitcast i8* %160 to i64*
  %162 = load i64, i64* %161, align 4
  %163 = icmp ne %String* %157, %35
  br i1 %163, label %condTrue__1, label %condContinue__1

condTrue__1:                                      ; preds = %body__1
  %164 = call %String* @__quantum__rt__string_concatenate(%String* %157, %String* %34)
  call void @__quantum__rt__string_update_reference_count(%String* %157, i32 -1)
  br label %condContinue__1

condContinue__1:                                  ; preds = %condTrue__1, %body__1
  %165 = phi %String* [ %164, %condTrue__1 ], [ %157, %body__1 ]
  %166 = call %String* @__quantum__rt__int_to_string(i64 %162)
  %167 = call %String* @__quantum__rt__string_concatenate(%String* %165, %String* %166)
  call void @__quantum__rt__string_update_reference_count(%String* %165, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %166, i32 -1)
  br label %exiting__1

exiting__1:                                       ; preds = %condContinue__1
  %168 = add i64 %158, 1
  br label %header__1

exit__1:                                          ; preds = %header__1
  %169 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @10, i32 0, i32 0))
  %170 = call %String* @__quantum__rt__string_concatenate(%String* %157, %String* %169)
  call void @__quantum__rt__string_update_reference_count(%String* %157, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %169, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %34, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %35, i32 -1)
  %171 = call %String* @__quantum__rt__string_concatenate(%String* %33, %String* %170)
  call void @__quantum__rt__string_update_reference_count(%String* %33, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %170, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %slice, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %everyOther, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %backwards, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %slice, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %everyOther, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %backwards, i32 -1)
  call void @__quantum__rt__fail(%String* %171)
  unreachable

header__2:                                        ; preds = %exiting__2, %then0__3
  %172 = phi %String* [ %55, %then0__3 ], [ %182, %exiting__2 ]
  %173 = phi i64 [ 0, %then0__3 ], [ %183, %exiting__2 ]
  %174 = icmp sle i64 %173, %57
  br i1 %174, label %body__2, label %exit__2

body__2:                                          ; preds = %header__2
  %175 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %slice, i64 %173)
  %176 = bitcast i8* %175 to i64*
  %177 = load i64, i64* %176, align 4
  %178 = icmp ne %String* %172, %55
  br i1 %178, label %condTrue__4, label %condContinue__4

condTrue__4:                                      ; preds = %body__2
  %179 = call %String* @__quantum__rt__string_concatenate(%String* %172, %String* %54)
  call void @__quantum__rt__string_update_reference_count(%String* %172, i32 -1)
  br label %condContinue__4

condContinue__4:                                  ; preds = %condTrue__4, %body__2
  %180 = phi %String* [ %179, %condTrue__4 ], [ %172, %body__2 ]
  %181 = call %String* @__quantum__rt__int_to_string(i64 %177)
  %182 = call %String* @__quantum__rt__string_concatenate(%String* %180, %String* %181)
  call void @__quantum__rt__string_update_reference_count(%String* %180, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %181, i32 -1)
  br label %exiting__2

exiting__2:                                       ; preds = %condContinue__4
  %183 = add i64 %173, 1
  br label %header__2

exit__2:                                          ; preds = %header__2
  %184 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @10, i32 0, i32 0))
  %185 = call %String* @__quantum__rt__string_concatenate(%String* %172, %String* %184)
  call void @__quantum__rt__string_update_reference_count(%String* %172, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %184, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %54, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %55, i32 -1)
  %186 = call %String* @__quantum__rt__string_concatenate(%String* %53, %String* %185)
  call void @__quantum__rt__string_update_reference_count(%String* %53, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %185, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %slice, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %everyOther, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %backwards, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %slice, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %everyOther, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %backwards, i32 -1)
  call void @__quantum__rt__fail(%String* %186)
  unreachable

header__3:                                        ; preds = %exiting__3, %then0__4
  %187 = phi %String* [ %66, %then0__4 ], [ %197, %exiting__3 ]
  %188 = phi i64 [ 0, %then0__4 ], [ %198, %exiting__3 ]
  %189 = icmp sle i64 %188, %67
  br i1 %189, label %body__3, label %exit__3

body__3:                                          ; preds = %header__3
  %190 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %everyOther, i64 %188)
  %191 = bitcast i8* %190 to i64*
  %192 = load i64, i64* %191, align 4
  %193 = icmp ne %String* %187, %66
  br i1 %193, label %condTrue__5, label %condContinue__5

condTrue__5:                                      ; preds = %body__3
  %194 = call %String* @__quantum__rt__string_concatenate(%String* %187, %String* %65)
  call void @__quantum__rt__string_update_reference_count(%String* %187, i32 -1)
  br label %condContinue__5

condContinue__5:                                  ; preds = %condTrue__5, %body__3
  %195 = phi %String* [ %194, %condTrue__5 ], [ %187, %body__3 ]
  %196 = call %String* @__quantum__rt__int_to_string(i64 %192)
  %197 = call %String* @__quantum__rt__string_concatenate(%String* %195, %String* %196)
  call void @__quantum__rt__string_update_reference_count(%String* %195, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %196, i32 -1)
  br label %exiting__3

exiting__3:                                       ; preds = %condContinue__5
  %198 = add i64 %188, 1
  br label %header__3

exit__3:                                          ; preds = %header__3
  %199 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @10, i32 0, i32 0))
  %200 = call %String* @__quantum__rt__string_concatenate(%String* %187, %String* %199)
  call void @__quantum__rt__string_update_reference_count(%String* %187, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %199, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %65, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %66, i32 -1)
  %201 = call %String* @__quantum__rt__string_concatenate(%String* %64, %String* %200)
  call void @__quantum__rt__string_update_reference_count(%String* %64, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %200, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %slice, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %everyOther, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %backwards, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %slice, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %everyOther, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %backwards, i32 -1)
  call void @__quantum__rt__fail(%String* %201)
  unreachable

header__4:                                        ; preds = %exiting__4, %then0__5
  %202 = phi %String* [ %86, %then0__5 ], [ %212, %exiting__4 ]
  %203 = phi i64 [ 0, %then0__5 ], [ %213, %exiting__4 ]
  %204 = icmp sle i64 %203, %88
  br i1 %204, label %body__4, label %exit__4

body__4:                                          ; preds = %header__4
  %205 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %everyOther, i64 %203)
  %206 = bitcast i8* %205 to i64*
  %207 = load i64, i64* %206, align 4
  %208 = icmp ne %String* %202, %86
  br i1 %208, label %condTrue__8, label %condContinue__8

condTrue__8:                                      ; preds = %body__4
  %209 = call %String* @__quantum__rt__string_concatenate(%String* %202, %String* %85)
  call void @__quantum__rt__string_update_reference_count(%String* %202, i32 -1)
  br label %condContinue__8

condContinue__8:                                  ; preds = %condTrue__8, %body__4
  %210 = phi %String* [ %209, %condTrue__8 ], [ %202, %body__4 ]
  %211 = call %String* @__quantum__rt__int_to_string(i64 %207)
  %212 = call %String* @__quantum__rt__string_concatenate(%String* %210, %String* %211)
  call void @__quantum__rt__string_update_reference_count(%String* %210, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %211, i32 -1)
  br label %exiting__4

exiting__4:                                       ; preds = %condContinue__8
  %213 = add i64 %203, 1
  br label %header__4

exit__4:                                          ; preds = %header__4
  %214 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @10, i32 0, i32 0))
  %215 = call %String* @__quantum__rt__string_concatenate(%String* %202, %String* %214)
  call void @__quantum__rt__string_update_reference_count(%String* %202, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %214, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %85, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %86, i32 -1)
  %216 = call %String* @__quantum__rt__string_concatenate(%String* %84, %String* %215)
  call void @__quantum__rt__string_update_reference_count(%String* %84, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %215, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %slice, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %everyOther, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %backwards, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %slice, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %everyOther, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %backwards, i32 -1)
  call void @__quantum__rt__fail(%String* %216)
  unreachable

header__5:                                        ; preds = %exiting__5, %then0__6
  %217 = phi %String* [ %97, %then0__6 ], [ %227, %exiting__5 ]
  %218 = phi i64 [ 0, %then0__6 ], [ %228, %exiting__5 ]
  %219 = icmp sle i64 %218, %98
  br i1 %219, label %body__5, label %exit__5

body__5:                                          ; preds = %header__5
  %220 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %backwards, i64 %218)
  %221 = bitcast i8* %220 to i64*
  %222 = load i64, i64* %221, align 4
  %223 = icmp ne %String* %217, %97
  br i1 %223, label %condTrue__9, label %condContinue__9

condTrue__9:                                      ; preds = %body__5
  %224 = call %String* @__quantum__rt__string_concatenate(%String* %217, %String* %96)
  call void @__quantum__rt__string_update_reference_count(%String* %217, i32 -1)
  br label %condContinue__9

condContinue__9:                                  ; preds = %condTrue__9, %body__5
  %225 = phi %String* [ %224, %condTrue__9 ], [ %217, %body__5 ]
  %226 = call %String* @__quantum__rt__int_to_string(i64 %222)
  %227 = call %String* @__quantum__rt__string_concatenate(%String* %225, %String* %226)
  call void @__quantum__rt__string_update_reference_count(%String* %225, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %226, i32 -1)
  br label %exiting__5

exiting__5:                                       ; preds = %condContinue__9
  %228 = add i64 %218, 1
  br label %header__5

exit__5:                                          ; preds = %header__5
  %229 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @10, i32 0, i32 0))
  %230 = call %String* @__quantum__rt__string_concatenate(%String* %217, %String* %229)
  call void @__quantum__rt__string_update_reference_count(%String* %217, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %229, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %96, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %97, i32 -1)
  %231 = call %String* @__quantum__rt__string_concatenate(%String* %95, %String* %230)
  call void @__quantum__rt__string_update_reference_count(%String* %95, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %230, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %slice, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %everyOther, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %backwards, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %slice, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %everyOther, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %backwards, i32 -1)
  call void @__quantum__rt__fail(%String* %231)
  unreachable

header__6:                                        ; preds = %exiting__6, %then0__7
  %232 = phi %String* [ %153, %then0__7 ], [ %242, %exiting__6 ]
  %233 = phi i64 [ 0, %then0__7 ], [ %243, %exiting__6 ]
  %234 = icmp sle i64 %233, %155
  br i1 %234, label %body__6, label %exit__6

body__6:                                          ; preds = %header__6
  %235 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %backwards, i64 %233)
  %236 = bitcast i8* %235 to i64*
  %237 = load i64, i64* %236, align 4
  %238 = icmp ne %String* %232, %153
  br i1 %238, label %condTrue__15, label %condContinue__15

condTrue__15:                                     ; preds = %body__6
  %239 = call %String* @__quantum__rt__string_concatenate(%String* %232, %String* %152)
  call void @__quantum__rt__string_update_reference_count(%String* %232, i32 -1)
  br label %condContinue__15

condContinue__15:                                 ; preds = %condTrue__15, %body__6
  %240 = phi %String* [ %239, %condTrue__15 ], [ %232, %body__6 ]
  %241 = call %String* @__quantum__rt__int_to_string(i64 %237)
  %242 = call %String* @__quantum__rt__string_concatenate(%String* %240, %String* %241)
  call void @__quantum__rt__string_update_reference_count(%String* %240, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %241, i32 -1)
  br label %exiting__6

exiting__6:                                       ; preds = %condContinue__15
  %243 = add i64 %233, 1
  br label %header__6

exit__6:                                          ; preds = %header__6
  %244 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @10, i32 0, i32 0))
  %245 = call %String* @__quantum__rt__string_concatenate(%String* %232, %String* %244)
  call void @__quantum__rt__string_update_reference_count(%String* %232, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %244, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %152, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %153, i32 -1)
  %246 = call %String* @__quantum__rt__string_concatenate(%String* %151, %String* %245)
  call void @__quantum__rt__string_update_reference_count(%String* %151, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %245, i32 -1)
  %247 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @16, i32 0, i32 0))
  %248 = call %String* @__quantum__rt__string_concatenate(%String* %246, %String* %247)
  call void @__quantum__rt__string_update_reference_count(%String* %246, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %247, i32 -1)
  %249 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @8, i32 0, i32 0))
  %250 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @9, i32 0, i32 0))
  call void @__quantum__rt__string_update_reference_count(%String* %250, i32 1)
  br label %header__7

header__7:                                        ; preds = %exiting__7, %exit__6
  %251 = phi %String* [ %250, %exit__6 ], [ %261, %exiting__7 ]
  %252 = phi i64 [ 0, %exit__6 ], [ %262, %exiting__7 ]
  %253 = icmp sle i64 %252, 5
  br i1 %253, label %body__7, label %exit__7

body__7:                                          ; preds = %header__7
  %254 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %array, i64 %252)
  %255 = bitcast i8* %254 to i64*
  %256 = load i64, i64* %255, align 4
  %257 = icmp ne %String* %251, %250
  br i1 %257, label %condTrue__16, label %condContinue__16

condTrue__16:                                     ; preds = %body__7
  %258 = call %String* @__quantum__rt__string_concatenate(%String* %251, %String* %249)
  call void @__quantum__rt__string_update_reference_count(%String* %251, i32 -1)
  br label %condContinue__16

condContinue__16:                                 ; preds = %condTrue__16, %body__7
  %259 = phi %String* [ %258, %condTrue__16 ], [ %251, %body__7 ]
  %260 = call %String* @__quantum__rt__int_to_string(i64 %256)
  %261 = call %String* @__quantum__rt__string_concatenate(%String* %259, %String* %260)
  call void @__quantum__rt__string_update_reference_count(%String* %259, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %260, i32 -1)
  br label %exiting__7

exiting__7:                                       ; preds = %condContinue__16
  %262 = add i64 %252, 1
  br label %header__7

exit__7:                                          ; preds = %header__7
  %263 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @10, i32 0, i32 0))
  %264 = call %String* @__quantum__rt__string_concatenate(%String* %251, %String* %263)
  call void @__quantum__rt__string_update_reference_count(%String* %251, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %263, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %249, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %250, i32 -1)
  %265 = call %String* @__quantum__rt__string_concatenate(%String* %248, %String* %264)
  call void @__quantum__rt__string_update_reference_count(%String* %248, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %264, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %slice, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %everyOther, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %backwards, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %slice, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %everyOther, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %backwards, i32 -1)
  call void @__quantum__rt__fail(%String* %265)
  unreachable
}

declare %Array* @__quantum__rt__array_create_1d(i32, i64)

declare i8* @__quantum__rt__array_get_element_ptr_1d(%Array*, i64)

declare void @__quantum__rt__array_update_alias_count(%Array*, i32)

declare %Array* @__quantum__rt__array_slice_1d(%Array*, %Range, i1)

declare %String* @__quantum__rt__int_to_string(i64)

declare void @__quantum__rt__array_update_reference_count(%Array*, i32)

declare i64 @__quantum__rt__array_get_size_1d(%Array*)

define internal void @Qir__Emission__ResultTest__body() {
entry:
  %one = call %Result* @__quantum__rt__result_get_one()
  %zero = call %Result* @__quantum__rt__result_get_zero()
  %0 = call %Result* @__quantum__rt__result_get_one()
  %1 = call i1 @__quantum__rt__result_equal(%Result* %0, %Result* %one)
  %2 = xor i1 %1, true
  br i1 %2, label %then0__1, label %continue__1

then0__1:                                         ; preds = %entry
  %3 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([35 x i8], [35 x i8]* @18, i32 0, i32 0))
  call void @__quantum__rt__fail(%String* %3)
  unreachable

continue__1:                                      ; preds = %entry
  %4 = call %Result* @__quantum__rt__result_get_zero()
  %5 = call i1 @__quantum__rt__result_equal(%Result* %4, %Result* %zero)
  %6 = xor i1 %5, true
  br i1 %6, label %then0__2, label %continue__2

then0__2:                                         ; preds = %continue__1
  %7 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([36 x i8], [36 x i8]* @19, i32 0, i32 0))
  call void @__quantum__rt__fail(%String* %7)
  unreachable

continue__2:                                      ; preds = %continue__1
  %8 = call i1 @__quantum__rt__result_equal(%Result* %one, %Result* %zero)
  br i1 %8, label %then0__3, label %continue__3

then0__3:                                         ; preds = %continue__2
  %9 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @20, i32 0, i32 0))
  call void @__quantum__rt__fail(%String* %9)
  unreachable

continue__3:                                      ; preds = %continue__2
  %10 = call %Result* @__quantum__rt__result_get_one()
  %11 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @21, i32 0, i32 0))
  %12 = call %String* @__quantum__rt__result_to_string(%Result* %10)
  %13 = call %String* @__quantum__rt__string_concatenate(%String* %11, %String* %12)
  call void @__quantum__rt__string_update_reference_count(%String* %11, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %12, i32 -1)
  %14 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @22, i32 0, i32 0))
  %15 = call %String* @__quantum__rt__string_concatenate(%String* %13, %String* %14)
  call void @__quantum__rt__string_update_reference_count(%String* %13, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %14, i32 -1)
  %16 = call %String* @__quantum__rt__result_to_string(%Result* %one)
  %17 = call %String* @__quantum__rt__string_concatenate(%String* %15, %String* %16)
  call void @__quantum__rt__string_update_reference_count(%String* %15, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %16, i32 -1)
  call void @__quantum__rt__message(%String* %17)
  %18 = call %Result* @__quantum__rt__result_get_zero()
  %19 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @23, i32 0, i32 0))
  %20 = call %String* @__quantum__rt__result_to_string(%Result* %18)
  %21 = call %String* @__quantum__rt__string_concatenate(%String* %19, %String* %20)
  call void @__quantum__rt__string_update_reference_count(%String* %19, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %20, i32 -1)
  %22 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @24, i32 0, i32 0))
  %23 = call %String* @__quantum__rt__string_concatenate(%String* %21, %String* %22)
  call void @__quantum__rt__string_update_reference_count(%String* %21, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %22, i32 -1)
  %24 = call %String* @__quantum__rt__result_to_string(%Result* %zero)
  %25 = call %String* @__quantum__rt__string_concatenate(%String* %23, %String* %24)
  call void @__quantum__rt__string_update_reference_count(%String* %23, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %24, i32 -1)
  call void @__quantum__rt__message(%String* %25)
  %26 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @25, i32 0, i32 0))
  call void @__quantum__rt__message(%String* %26)
  call void @__quantum__rt__string_update_reference_count(%String* %17, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %25, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %26, i32 -1)
  ret void
}

declare %Result* @__quantum__rt__result_get_one()

declare %Result* @__quantum__rt__result_get_zero()

declare i1 @__quantum__rt__result_equal(%Result*, %Result*)

declare %String* @__quantum__rt__result_to_string(%Result*)

define internal void @Qir__Emission__TupleTest__body() {
entry:
  %0 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ i64, %String* }* getelementptr ({ i64, %String* }, { i64, %String* }* null, i32 1) to i64))
  %a = bitcast %Tuple* %0 to { i64, %String* }*
  %1 = getelementptr inbounds { i64, %String* }, { i64, %String* }* %a, i32 0, i32 0
  %2 = getelementptr inbounds { i64, %String* }, { i64, %String* }* %a, i32 0, i32 1
  %c = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @1, i32 0, i32 0))
  store i64 1, i64* %1, align 4
  store %String* %c, %String** %2, align 8
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %0, i32 1)
  %3 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %String*, i64 }* getelementptr ({ %String*, i64 }, { %String*, i64 }* null, i32 1) to i64))
  %b = bitcast %Tuple* %3 to { %String*, i64 }*
  %4 = getelementptr inbounds { %String*, i64 }, { %String*, i64 }* %b, i32 0, i32 0
  %5 = getelementptr inbounds { %String*, i64 }, { %String*, i64 }* %b, i32 0, i32 1
  %d = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @0, i32 0, i32 0))
  store %String* %d, %String** %4, align 8
  store i64 1, i64* %5, align 4
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %3, i32 1)
  %6 = call %String* @__quantum__rt__string_concatenate(%String* %d, %String* %c)
  %7 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @28, i32 0, i32 0))
  %8 = call i1 @__quantum__rt__string_equal(%String* %6, %String* %7)
  %9 = xor i1 %8, true
  call void @__quantum__rt__string_update_reference_count(%String* %6, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %7, i32 -1)
  br i1 %9, label %then0__1, label %continue__1

then0__1:                                         ; preds = %entry
  %10 = call %String* @__quantum__rt__string_concatenate(%String* %d, %String* %c)
  %11 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([41 x i8], [41 x i8]* @29, i32 0, i32 0))
  %12 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @30, i32 0, i32 0))
  %13 = call %String* @__quantum__rt__string_concatenate(%String* %12, %String* %10)
  %14 = call %String* @__quantum__rt__string_concatenate(%String* %13, %String* %12)
  call void @__quantum__rt__string_update_reference_count(%String* %13, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %12, i32 -1)
  %15 = call %String* @__quantum__rt__string_concatenate(%String* %11, %String* %14)
  call void @__quantum__rt__string_update_reference_count(%String* %11, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %14, i32 -1)
  %16 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @31, i32 0, i32 0))
  %17 = call %String* @__quantum__rt__string_concatenate(%String* %15, %String* %16)
  call void @__quantum__rt__string_update_reference_count(%String* %15, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %16, i32 -1)
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %0, i32 -1)
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %3, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %10, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %c, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %0, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %d, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %3, i32 -1)
  call void @__quantum__rt__fail(%String* %17)
  unreachable

continue__1:                                      ; preds = %entry
  %18 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @32, i32 0, i32 0))
  call void @__quantum__rt__message(%String* %18)
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %0, i32 -1)
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %3, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %c, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %0, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %d, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %3, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %18, i32 -1)
  ret void
}

declare i1 @__quantum__rt__string_equal(%String*, %String*)

define void @Qir__Emission__BasicTest__Interop() #0 {
entry:
  call void @Qir__Emission__BasicTest__body()
  ret void
}

define void @Qir__Emission__BasicTest() #1 {
entry:
  call void @Qir__Emission__BasicTest__body()
  %0 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @33, i32 0, i32 0))
  call void @__quantum__rt__message(%String* %0)
  call void @__quantum__rt__string_update_reference_count(%String* %0, i32 -1)
  ret void
}

define i8* @Qir__Emission__ConcatTest__Interop() #0 {
entry:
  %0 = call %String* @Qir__Emission__ConcatTest__body()
  %1 = call i32 @__quantum__rt__string_get_length(%String* %0)
  %2 = sext i32 %1 to i64
  %3 = call i8* @__quantum__rt__string_get_data(%String* %0)
  %4 = call i8* @__quantum__rt__memory_allocate(i64 %2)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %4, i8* %3, i64 %2, i1 false)
  call void @__quantum__rt__string_update_reference_count(%String* %0, i32 -1)
  ret i8* %4
}

declare i32 @__quantum__rt__string_get_length(%String*)

declare i8* @__quantum__rt__string_get_data(%String*)

declare i8* @__quantum__rt__memory_allocate(i64)

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #2

define void @Qir__Emission__ConcatTest() #1 {
entry:
  %0 = call %String* @Qir__Emission__ConcatTest__body()
  %1 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @30, i32 0, i32 0))
  %2 = call %String* @__quantum__rt__string_concatenate(%String* %1, %String* %0)
  %3 = call %String* @__quantum__rt__string_concatenate(%String* %2, %String* %1)
  call void @__quantum__rt__string_update_reference_count(%String* %2, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %1, i32 -1)
  call void @__quantum__rt__message(%String* %3)
  call void @__quantum__rt__string_update_reference_count(%String* %0, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %3, i32 -1)
  ret void
}

define void @Qir__Emission__CallableTest__Interop(i8 %shouldFail) #0 {
entry:
  %0 = trunc i8 %shouldFail to i1
  call void @Qir__Emission__CallableTest__body(i1 %0)
  ret void
}

define void @Qir__Emission__CallableTest(i8 %shouldFail) #1 {
entry:
  %0 = trunc i8 %shouldFail to i1
  call void @Qir__Emission__CallableTest__body(i1 %0)
  %1 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @33, i32 0, i32 0))
  call void @__quantum__rt__message(%String* %1)
  call void @__quantum__rt__string_update_reference_count(%String* %1, i32 -1)
  ret void
}

define void @Qir__Emission__ResultTest__Interop() #0 {
entry:
  call void @Qir__Emission__ResultTest__body()
  ret void
}

define void @Qir__Emission__ResultTest() #1 {
entry:
  call void @Qir__Emission__ResultTest__body()
  %0 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @33, i32 0, i32 0))
  call void @__quantum__rt__message(%String* %0)
  call void @__quantum__rt__string_update_reference_count(%String* %0, i32 -1)
  ret void
}

define void @Qir__Emission__RangeTest__Interop() #0 {
entry:
  call void @Qir__Emission__RangeTest__body()
  ret void
}

define void @Qir__Emission__RangeTest() #1 {
entry:
  call void @Qir__Emission__RangeTest__body()
  %0 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @33, i32 0, i32 0))
  call void @__quantum__rt__message(%String* %0)
  call void @__quantum__rt__string_update_reference_count(%String* %0, i32 -1)
  ret void
}

define void @Qir__Emission__CustomFail__Interop() #0 {
entry:
  call void @Qir__Emission__CustomFail__body()
  ret void
}

define void @Qir__Emission__CustomFail() #1 {
entry:
  call void @Qir__Emission__CustomFail__body()
  %0 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @33, i32 0, i32 0))
  call void @__quantum__rt__message(%String* %0)
  call void @__quantum__rt__string_update_reference_count(%String* %0, i32 -1)
  ret void
}

define void @Qir__Emission__TupleTest__Interop() #0 {
entry:
  call void @Qir__Emission__TupleTest__body()
  ret void
}

define void @Qir__Emission__TupleTest() #1 {
entry:
  call void @Qir__Emission__TupleTest__body()
  %0 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @33, i32 0, i32 0))
  call void @__quantum__rt__message(%String* %0)
  call void @__quantum__rt__string_update_reference_count(%String* %0, i32 -1)
  ret void
}

attributes #0 = { "InteropFriendly" }
attributes #1 = { "EntryPoint" }
attributes #2 = { argmemonly nounwind willreturn }
