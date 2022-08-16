; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%Qubit = type opaque
%Result = type opaque

@nl = internal constant [2 x i8] c"\0A\00"
@frmt_spec = internal constant [4 x i8] c"%i \00"
@q = global [3 x %Qubit*] undef
@a = global [3 x i64] undef

declare i8* @malloc(i64)

declare void @free(i8*)

declare i32 @printf(i8*, ...)

declare i1 @__quantum__rt__result_equal(%Result*, %Result*)

declare %Result* @__quantum__rt__result_get_one()

declare void @__quantum__qis__t__adj(%Qubit*)

declare void @__quantum__rt__qubit_release(%Qubit*)

declare %Qubit* @__quantum__rt__qubit_allocate()

declare void @__quantum__qis__t__body(%Qubit*)

declare void @__quantum__qis__s__body(%Qubit*)

declare void @__quantum__qis__x__body(%Qubit*)

declare void @__quantum__qis__h__body(%Qubit*)

declare void @__quantum__qis__cnot(%Qubit*, %Qubit*)

declare void @__quantum__qis__reset(%Qubit*)

declare %Result* @__quantum__qis__measure(%Qubit*)

define void @toffoli(%Qubit** %0, %Qubit** %1, i64 %2, i64 %3, i64 %4, %Qubit** %5, %Qubit** %6, i64 %7, i64 %8, i64 %9, %Qubit** %10, %Qubit** %11, i64 %12, i64 %13, i64 %14) !dbg !3 {
  %16 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } undef, %Qubit** %0, 0, !dbg !7
  %17 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %16, %Qubit** %1, 1, !dbg !7
  %18 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %17, i64 %2, 2, !dbg !7
  %19 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %18, i64 %3, 3, 0, !dbg !7
  %20 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %19, i64 %4, 4, 0, !dbg !7
  %21 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } undef, %Qubit** %5, 0, !dbg !9
  %22 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %21, %Qubit** %6, 1, !dbg !9
  %23 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %22, i64 %7, 2, !dbg !9
  %24 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %23, i64 %8, 3, 0, !dbg !9
  %25 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %24, i64 %9, 4, 0, !dbg !9
  %26 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } undef, %Qubit** %10, 0, !dbg !10
  %27 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %26, %Qubit** %11, 1, !dbg !10
  %28 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %27, i64 %12, 2, !dbg !10
  %29 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %28, i64 %13, 3, 0, !dbg !10
  %30 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %29, i64 %14, 4, 0, !dbg !10
  %31 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 1, !dbg !11
  %32 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 2, !dbg !11
  %33 = add i64 %32, 0, !dbg !11
  %34 = getelementptr %Qubit*, %Qubit** %31, i64 %33, !dbg !11
  %35 = load %Qubit*, %Qubit** %34, align 8, !dbg !11
  call void @__quantum__qis__h__body(%Qubit* %35), !dbg !11
  %36 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %25, 1, !dbg !14
  %37 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %25, 2, !dbg !14
  %38 = add i64 %37, 0, !dbg !14
  %39 = getelementptr %Qubit*, %Qubit** %36, i64 %38, !dbg !14
  %40 = load %Qubit*, %Qubit** %39, align 8, !dbg !14
  %41 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 1, !dbg !14
  %42 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 2, !dbg !14
  %43 = add i64 %42, 0, !dbg !14
  %44 = getelementptr %Qubit*, %Qubit** %41, i64 %43, !dbg !14
  %45 = load %Qubit*, %Qubit** %44, align 8, !dbg !14
  call void @__quantum__qis__cnot(%Qubit* %40, %Qubit* %45), !dbg !14
  %46 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 1, !dbg !15
  %47 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 2, !dbg !15
  %48 = add i64 %47, 0, !dbg !15
  %49 = getelementptr %Qubit*, %Qubit** %46, i64 %48, !dbg !15
  %50 = load %Qubit*, %Qubit** %49, align 8, !dbg !15
  call void @__quantum__qis__t__adj(%Qubit* %50), !dbg !15
  %51 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %20, 1, !dbg !16
  %52 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %20, 2, !dbg !16
  %53 = add i64 %52, 0, !dbg !16
  %54 = getelementptr %Qubit*, %Qubit** %51, i64 %53, !dbg !16
  %55 = load %Qubit*, %Qubit** %54, align 8, !dbg !16
  %56 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 1, !dbg !16
  %57 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 2, !dbg !16
  %58 = add i64 %57, 0, !dbg !16
  %59 = getelementptr %Qubit*, %Qubit** %56, i64 %58, !dbg !16
  %60 = load %Qubit*, %Qubit** %59, align 8, !dbg !16
  call void @__quantum__qis__cnot(%Qubit* %55, %Qubit* %60), !dbg !16
  %61 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 1, !dbg !17
  %62 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 2, !dbg !17
  %63 = add i64 %62, 0, !dbg !17
  %64 = getelementptr %Qubit*, %Qubit** %61, i64 %63, !dbg !17
  %65 = load %Qubit*, %Qubit** %64, align 8, !dbg !17
  call void @__quantum__qis__t__body(%Qubit* %65), !dbg !17
  %66 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %25, 1, !dbg !18
  %67 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %25, 2, !dbg !18
  %68 = add i64 %67, 0, !dbg !18
  %69 = getelementptr %Qubit*, %Qubit** %66, i64 %68, !dbg !18
  %70 = load %Qubit*, %Qubit** %69, align 8, !dbg !18
  %71 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 1, !dbg !18
  %72 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 2, !dbg !18
  %73 = add i64 %72, 0, !dbg !18
  %74 = getelementptr %Qubit*, %Qubit** %71, i64 %73, !dbg !18
  %75 = load %Qubit*, %Qubit** %74, align 8, !dbg !18
  call void @__quantum__qis__cnot(%Qubit* %70, %Qubit* %75), !dbg !18
  %76 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 1, !dbg !19
  %77 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 2, !dbg !19
  %78 = add i64 %77, 0, !dbg !19
  %79 = getelementptr %Qubit*, %Qubit** %76, i64 %78, !dbg !19
  %80 = load %Qubit*, %Qubit** %79, align 8, !dbg !19
  call void @__quantum__qis__t__adj(%Qubit* %80), !dbg !19
  %81 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %20, 1, !dbg !20
  %82 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %20, 2, !dbg !20
  %83 = add i64 %82, 0, !dbg !20
  %84 = getelementptr %Qubit*, %Qubit** %81, i64 %83, !dbg !20
  %85 = load %Qubit*, %Qubit** %84, align 8, !dbg !20
  %86 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 1, !dbg !20
  %87 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 2, !dbg !20
  %88 = add i64 %87, 0, !dbg !20
  %89 = getelementptr %Qubit*, %Qubit** %86, i64 %88, !dbg !20
  %90 = load %Qubit*, %Qubit** %89, align 8, !dbg !20
  call void @__quantum__qis__cnot(%Qubit* %85, %Qubit* %90), !dbg !20
  %91 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %25, 1, !dbg !21
  %92 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %25, 2, !dbg !21
  %93 = add i64 %92, 0, !dbg !21
  %94 = getelementptr %Qubit*, %Qubit** %91, i64 %93, !dbg !21
  %95 = load %Qubit*, %Qubit** %94, align 8, !dbg !21
  call void @__quantum__qis__t__adj(%Qubit* %95), !dbg !21
  %96 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 1, !dbg !22
  %97 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 2, !dbg !22
  %98 = add i64 %97, 0, !dbg !22
  %99 = getelementptr %Qubit*, %Qubit** %96, i64 %98, !dbg !22
  %100 = load %Qubit*, %Qubit** %99, align 8, !dbg !22
  call void @__quantum__qis__t__body(%Qubit* %100), !dbg !22
  %101 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 1, !dbg !23
  %102 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %30, 2, !dbg !23
  %103 = add i64 %102, 0, !dbg !23
  %104 = getelementptr %Qubit*, %Qubit** %101, i64 %103, !dbg !23
  %105 = load %Qubit*, %Qubit** %104, align 8, !dbg !23
  call void @__quantum__qis__h__body(%Qubit* %105), !dbg !23
  %106 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %20, 1, !dbg !24
  %107 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %20, 2, !dbg !24
  %108 = add i64 %107, 0, !dbg !24
  %109 = getelementptr %Qubit*, %Qubit** %106, i64 %108, !dbg !24
  %110 = load %Qubit*, %Qubit** %109, align 8, !dbg !24
  %111 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %25, 1, !dbg !24
  %112 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %25, 2, !dbg !24
  %113 = add i64 %112, 0, !dbg !24
  %114 = getelementptr %Qubit*, %Qubit** %111, i64 %113, !dbg !24
  %115 = load %Qubit*, %Qubit** %114, align 8, !dbg !24
  call void @__quantum__qis__cnot(%Qubit* %110, %Qubit* %115), !dbg !24
  %116 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %25, 1, !dbg !25
  %117 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %25, 2, !dbg !25
  %118 = add i64 %117, 0, !dbg !25
  %119 = getelementptr %Qubit*, %Qubit** %116, i64 %118, !dbg !25
  %120 = load %Qubit*, %Qubit** %119, align 8, !dbg !25
  call void @__quantum__qis__t__adj(%Qubit* %120), !dbg !25
  %121 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %20, 1, !dbg !26
  %122 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %20, 2, !dbg !26
  %123 = add i64 %122, 0, !dbg !26
  %124 = getelementptr %Qubit*, %Qubit** %121, i64 %123, !dbg !26
  %125 = load %Qubit*, %Qubit** %124, align 8, !dbg !26
  %126 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %25, 1, !dbg !26
  %127 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %25, 2, !dbg !26
  %128 = add i64 %127, 0, !dbg !26
  %129 = getelementptr %Qubit*, %Qubit** %126, i64 %128, !dbg !26
  %130 = load %Qubit*, %Qubit** %129, align 8, !dbg !26
  call void @__quantum__qis__cnot(%Qubit* %125, %Qubit* %130), !dbg !26
  %131 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %20, 1, !dbg !27
  %132 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %20, 2, !dbg !27
  %133 = add i64 %132, 0, !dbg !27
  %134 = getelementptr %Qubit*, %Qubit** %131, i64 %133, !dbg !27
  %135 = load %Qubit*, %Qubit** %134, align 8, !dbg !27
  call void @__quantum__qis__t__body(%Qubit* %135), !dbg !27
  %136 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %25, 1, !dbg !28
  %137 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %25, 2, !dbg !28
  %138 = add i64 %137, 0, !dbg !28
  %139 = getelementptr %Qubit*, %Qubit** %136, i64 %138, !dbg !28
  %140 = load %Qubit*, %Qubit** %139, align 8, !dbg !28
  call void @__quantum__qis__s__body(%Qubit* %140), !dbg !28
  ret void, !dbg !29
}

define void @test_main() !dbg !30 {
  br label %1, !dbg !31

1:                                                ; preds = %4, %0
  %2 = phi i64 [ 0, %0 ], [ %7, %4 ]
  %3 = icmp slt i64 %2, 3, !dbg !31
  br i1 %3, label %4, label %8, !dbg !31

4:                                                ; preds = %1
  %5 = call %Qubit* @__quantum__rt__qubit_allocate(), !dbg !31
  %6 = getelementptr %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 0), i64 %2, !dbg !31
  store %Qubit* %5, %Qubit** %6, align 8, !dbg !31
  %7 = add i64 %2, 1, !dbg !31
  br label %1, !dbg !31

8:                                                ; preds = %1
  %9 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 1), align 8, !dbg !33
  call void @__quantum__qis__x__body(%Qubit* %9), !dbg !33
  %10 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 2), align 8, !dbg !34
  call void @__quantum__qis__x__body(%Qubit* %10), !dbg !34
  call void @toffoli(%Qubit** inttoptr (i64 3735928559 to %Qubit**), %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 0), i64 1, i64 1, i64 1, %Qubit** inttoptr (i64 3735928559 to %Qubit**), %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 0), i64 2, i64 1, i64 1, %Qubit** inttoptr (i64 3735928559 to %Qubit**), %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 0), i64 0, i64 1, i64 1), !dbg !35
  %11 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 2), align 8, !dbg !36
  %12 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 1), align 8, !dbg !36
  call void @__quantum__qis__cnot(%Qubit* %11, %Qubit* %12), !dbg !36
  %13 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 2), align 8, !dbg !37
  call void @__quantum__qis__x__body(%Qubit* %13), !dbg !37
  br label %14, !dbg !38

14:                                               ; preds = %26, %8
  %15 = phi i64 [ 0, %8 ], [ %78, %26 ]
  %16 = icmp slt i64 %15, 3, !dbg !38
  br i1 %16, label %17, label %79, !dbg !38

17:                                               ; preds = %14
  %18 = call i8* @malloc(i64 ptrtoint (i64* getelementptr (i64, i64* null, i64 1) to i64)), !dbg !38
  %19 = bitcast i8* %18 to i64*, !dbg !38
  br label %20, !dbg !38

20:                                               ; preds = %23, %17
  %21 = phi i64 [ 0, %17 ], [ %25, %23 ]
  %22 = icmp slt i64 %21, 1, !dbg !38
  br i1 %22, label %23, label %26, !dbg !38

23:                                               ; preds = %20
  %24 = getelementptr i64, i64* %19, i64 %21, !dbg !38
  store i64 0, i64* %24, align 4, !dbg !38
  %25 = add i64 %21, 1, !dbg !38
  br label %20, !dbg !38

26:                                               ; preds = %20
  %27 = insertvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } undef, i64* %19, 0, !dbg !38
  %28 = insertvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } %27, i64* %19, 1, !dbg !38
  %29 = insertvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } %28, i64 0, 2, !dbg !38
  %30 = insertvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } %29, i64 1, 3, 0, !dbg !38
  %31 = insertvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } %30, i64 1, 4, 0, !dbg !38
  %32 = extractvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } %31, 1, !dbg !38
  %33 = getelementptr i64, i64* %32, i64 0, !dbg !38
  store i64 %15, i64* %33, align 4, !dbg !38
  %34 = extractvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } %31, 1, !dbg !39
  %35 = getelementptr i64, i64* %34, i64 0, !dbg !39
  %36 = load i64, i64* %35, align 4, !dbg !39
  %37 = mul i64 %36, 1, !dbg !40
  %38 = add i64 0, %37, !dbg !40
  %39 = insertvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } { i64* inttoptr (i64 3735928559 to i64*), i64* getelementptr inbounds ([3 x i64], [3 x i64]* @a, i64 0, i64 0), i64 undef, [1 x i64] undef, [1 x i64] undef }, i64 %38, 2, !dbg !40
  %40 = insertvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } %39, i64 1, 3, 0, !dbg !40
  %41 = insertvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } %40, i64 1, 4, 0, !dbg !40
  %42 = extractvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } %31, 1, !dbg !41
  %43 = getelementptr i64, i64* %42, i64 0, !dbg !41
  %44 = load i64, i64* %43, align 4, !dbg !41
  %45 = mul i64 %44, 1, !dbg !42
  %46 = add i64 0, %45, !dbg !42
  %47 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } { %Qubit** inttoptr (i64 3735928559 to %Qubit**), %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 0), i64 undef, [1 x i64] undef, [1 x i64] undef }, i64 %46, 2, !dbg !42
  %48 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %47, i64 1, 3, 0, !dbg !42
  %49 = insertvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %48, i64 1, 4, 0, !dbg !42
  %50 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %49, 1, !dbg !43
  %51 = extractvalue { %Qubit**, %Qubit**, i64, [1 x i64], [1 x i64] } %49, 2, !dbg !43
  %52 = add i64 %51, 0, !dbg !43
  %53 = getelementptr %Qubit*, %Qubit** %50, i64 %52, !dbg !43
  %54 = load %Qubit*, %Qubit** %53, align 8, !dbg !43
  %55 = call %Result* @__quantum__qis__measure(%Qubit* %54), !dbg !43
  %56 = call %Result* @__quantum__rt__result_get_one(), !dbg !43
  %57 = call i1 @__quantum__rt__result_equal(%Result* %55, %Result* %56), !dbg !43
  %58 = zext i1 %57 to i64, !dbg !43
  %59 = extractvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } %41, 1, !dbg !40
  %60 = extractvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } %41, 2, !dbg !40
  %61 = add i64 %60, 0, !dbg !40
  %62 = getelementptr i64, i64* %59, i64 %61, !dbg !40
  store i64 %58, i64* %62, align 4, !dbg !40
  %63 = extractvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } %31, 1, !dbg !44
  %64 = getelementptr i64, i64* %63, i64 0, !dbg !44
  %65 = load i64, i64* %64, align 4, !dbg !44
  %66 = mul i64 %65, 1, !dbg !45
  %67 = add i64 0, %66, !dbg !45
  %68 = insertvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } { i64* inttoptr (i64 3735928559 to i64*), i64* getelementptr inbounds ([3 x i64], [3 x i64]* @a, i64 0, i64 0), i64 undef, [1 x i64] undef, [1 x i64] undef }, i64 %67, 2, !dbg !45
  %69 = insertvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } %68, i64 1, 3, 0, !dbg !45
  %70 = insertvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } %69, i64 1, 4, 0, !dbg !45
  %71 = extractvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } %70, 1, !dbg !45
  %72 = extractvalue { i64*, i64*, i64, [1 x i64], [1 x i64] } %70, 2, !dbg !45
  %73 = add i64 %72, 0, !dbg !45
  %74 = getelementptr i64, i64* %71, i64 %73, !dbg !45
  %75 = load i64, i64* %74, align 4, !dbg !45
  %76 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), i64 %75), !dbg !46
  %77 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @nl, i64 0, i64 0)), !dbg !46
  %78 = add i64 %15, 1, !dbg !38
  br label %14, !dbg !38

79:                                               ; preds = %14
  br label %80, !dbg !31

80:                                               ; preds = %83, %79
  %81 = phi i64 [ 0, %79 ], [ %86, %83 ]
  %82 = icmp slt i64 %81, 3, !dbg !31
  br i1 %82, label %83, label %87, !dbg !31

83:                                               ; preds = %80
  %84 = getelementptr %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 0), i64 %81, !dbg !31
  %85 = load %Qubit*, %Qubit** %84, align 8, !dbg !31
  call void @__quantum__rt__qubit_release(%Qubit* %85), !dbg !31
  %86 = add i64 %81, 1, !dbg !31
  br label %80, !dbg !31

87:                                               ; preds = %80
  ret void, !dbg !47
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "toffoli", linkageName: "toffoli", scope: null, file: !4, line: 14, type: !5, scopeLine: 14, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "../tests/out2.mlir", directory: "/Users/huazhelou/Documents/quantum/llvm/isqv2/mlir/build")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 15, column: 8, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 15, column: 67, scope: !8)
!10 = !DILocation(line: 15, column: 126, scope: !8)
!11 = !DILocation(line: 16, column: 5, scope: !12)
!12 = !DILexicalBlockFile(scope: !3, file: !13, discriminator: 0)
!13 = !DIFile(filename: "main.isq", directory: "/Users/huazhelou/Documents/quantum/llvm/isqv2/mlir/build")
!14 = !DILocation(line: 17, column: 5, scope: !12)
!15 = !DILocation(line: 18, column: 5, scope: !12)
!16 = !DILocation(line: 19, column: 5, scope: !12)
!17 = !DILocation(line: 20, column: 5, scope: !12)
!18 = !DILocation(line: 21, column: 5, scope: !12)
!19 = !DILocation(line: 22, column: 5, scope: !12)
!20 = !DILocation(line: 23, column: 5, scope: !12)
!21 = !DILocation(line: 24, column: 5, scope: !12)
!22 = !DILocation(line: 25, column: 5, scope: !12)
!23 = !DILocation(line: 26, column: 5, scope: !12)
!24 = !DILocation(line: 27, column: 5, scope: !12)
!25 = !DILocation(line: 28, column: 5, scope: !12)
!26 = !DILocation(line: 29, column: 5, scope: !12)
!27 = !DILocation(line: 30, column: 5, scope: !12)
!28 = !DILocation(line: 31, column: 5, scope: !12)
!29 = !DILocation(line: 15, column: 1, scope: !12)
!30 = distinct !DISubprogram(name: "test_main", linkageName: "test_main", scope: null, file: !4, line: 117, type: !5, scopeLine: 117, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!31 = !DILocation(line: 12, column: 6, scope: !32)
!32 = !DILexicalBlockFile(scope: !30, file: !13, discriminator: 0)
!33 = !DILocation(line: 36, column: 5, scope: !32)
!34 = !DILocation(line: 37, column: 5, scope: !32)
!35 = !DILocation(line: 40, column: 5, scope: !32)
!36 = !DILocation(line: 41, column: 5, scope: !32)
!37 = !DILocation(line: 42, column: 5, scope: !32)
!38 = !DILocation(line: 43, column: 5, scope: !32)
!39 = !DILocation(line: 44, column: 11, scope: !32)
!40 = !DILocation(line: 44, column: 9, scope: !32)
!41 = !DILocation(line: 44, column: 20, scope: !32)
!42 = !DILocation(line: 44, column: 18, scope: !32)
!43 = !DILocation(line: 44, column: 16, scope: !32)
!44 = !DILocation(line: 45, column: 17, scope: !32)
!45 = !DILocation(line: 45, column: 15, scope: !32)
!46 = !DILocation(line: 45, column: 9, scope: !32)
!47 = !DILocation(line: 34, column: 1, scope: !32)

