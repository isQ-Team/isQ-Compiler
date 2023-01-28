; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%Qubit = type opaque
%Result = type opaque

@nl = internal constant [2 x i8] c"\0A\00"
@frmt_spec = internal constant [4 x i8] c"%i \00"
@q = global [2 x %Qubit*] undef
@a = global [2 x i64] undef

declare i8* @malloc(i64)

declare void @free(i8*)

declare i32 @printf(i8*, ...)

declare i1 @__quantum__rt__result_equal(%Result*, %Result*)

declare %Result* @__quantum__rt__result_get_one()

declare void @__quantum__rt__qubit_release(%Qubit*)

declare %Qubit* @__quantum__rt__qubit_allocate()

declare void @__quantum__qis__cnot(%Qubit*, %Qubit*)

declare void @__quantum__qis__h__body(%Qubit*)

declare void @__quantum__rt__reset(%Qubit*)

declare %Result* @__quantum__qis__measure(%Qubit*)

define void @bell_main() !dbg !3 {
  br label %1, !dbg !7

1:                                                ; preds = %4, %0
  %2 = phi i64 [ 0, %0 ], [ %7, %4 ]
  %3 = icmp slt i64 %2, 2, !dbg !7
  br i1 %3, label %4, label %8, !dbg !7

4:                                                ; preds = %1
  %5 = call %Qubit* @__quantum__rt__qubit_allocate(), !dbg !7
  %6 = getelementptr %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 0), i64 %2, !dbg !7
  store %Qubit* %5, %Qubit** %6, align 8, !dbg !7
  %7 = add i64 %2, 1, !dbg !7
  br label %1, !dbg !7

8:                                                ; preds = %1
  %9 = load %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !10
  call void @__quantum__qis__h__body(%Qubit* %9), !dbg !10
  %10 = load %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !11
  %11 = load %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 1), align 8, !dbg !11
  call void @__quantum__qis__cnot(%Qubit* %10, %Qubit* %11), !dbg !11
  %12 = load %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !12
  %13 = call %Result* @__quantum__qis__measure(%Qubit* %12), !dbg !12
  %14 = call %Result* @__quantum__rt__result_get_one(), !dbg !12
  %15 = call i1 @__quantum__rt__result_equal(%Result* %13, %Result* %14), !dbg !12
  %16 = sext i1 %15 to i64, !dbg !12
  store i64 %16, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @a, i64 0, i64 0), align 4, !dbg !13
  %17 = load %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 1), align 8, !dbg !14
  %18 = call %Result* @__quantum__qis__measure(%Qubit* %17), !dbg !14
  %19 = call %Result* @__quantum__rt__result_get_one(), !dbg !14
  %20 = call i1 @__quantum__rt__result_equal(%Result* %18, %Result* %19), !dbg !14
  %21 = sext i1 %20 to i64, !dbg !14
  store i64 %21, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @a, i64 0, i64 1), align 4, !dbg !15
  %22 = load i64, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @a, i64 0, i64 0), align 4, !dbg !16
  %23 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), i64 %22), !dbg !17
  %24 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @nl, i64 0, i64 0)), !dbg !17
  %25 = load i64, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @a, i64 0, i64 1), align 4, !dbg !18
  %26 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), i64 %25), !dbg !19
  %27 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @nl, i64 0, i64 0)), !dbg !19
  br label %28, !dbg !7

28:                                               ; preds = %31, %8
  %29 = phi i64 [ 0, %8 ], [ %34, %31 ]
  %30 = icmp slt i64 %29, 2, !dbg !7
  br i1 %30, label %31, label %35, !dbg !7

31:                                               ; preds = %28
  %32 = getelementptr %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 0), i64 %29, !dbg !7
  %33 = load %Qubit*, %Qubit** %32, align 8, !dbg !7
  call void @__quantum__rt__qubit_release(%Qubit* %33), !dbg !7
  %34 = add i64 %29, 1, !dbg !7
  br label %28, !dbg !7

35:                                               ; preds = %28
  ret void, !dbg !20
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "main", linkageName: "main", scope: null, file: !4, line: 11, type: !5, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "../tests/out.mlir", directory: "/Users/huazhelou/Documents/quantum/llvm/isqv2/mlir/build")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 4, column: 6, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !9, discriminator: 0)
!9 = !DIFile(filename: "main.isq", directory: "/Users/huazhelou/Documents/quantum/llvm/isqv2/mlir/build")
!10 = !DILocation(line: 8, column: 9, scope: !8)
!11 = !DILocation(line: 9, column: 9, scope: !8)
!12 = !DILocation(line: 10, column: 16, scope: !8)
!13 = !DILocation(line: 10, column: 9, scope: !8)
!14 = !DILocation(line: 11, column: 16, scope: !8)
!15 = !DILocation(line: 11, column: 9, scope: !8)
!16 = !DILocation(line: 12, column: 15, scope: !8)
!17 = !DILocation(line: 12, column: 9, scope: !8)
!18 = !DILocation(line: 13, column: 15, scope: !8)
!19 = !DILocation(line: 13, column: 9, scope: !8)
!20 = !DILocation(line: 6, column: 1, scope: !8)

