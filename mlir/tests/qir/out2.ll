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

declare void @__quantum__qir__h(%Qubit*)

declare void @__quantum__qir__cnot(%Qubit*, %Qubit*)

declare void @__quantum__qir__u3(double, double, double, %Qubit*)

declare void @__quantum__qir__gphase(double)

define void @__quantum__qir__rs(%Qubit* %0, %Qubit* %1) !dbg !3 {
  call void @__quantum__qir__gphase(double 0x3FE0C15237AB6B20), !dbg !7
  call void @__quantum__qir__u3(double 0x3FF921FB54B4C996, double 0xBFF921FB54442D18, double 0x400921FB54442D18, %Qubit* %1), !dbg !7
  call void @__quantum__qir__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFF921FB54442D18, %Qubit* %0), !dbg !7
  call void @__quantum__qir__cnot(%Qubit* %1, %Qubit* %0), !dbg !7
  call void @__quantum__qir__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FF921FB54442D18, %Qubit* %0), !dbg !7
  call void @__quantum__qir__cnot(%Qubit* %1, %Qubit* %0), !dbg !7
  call void @__quantum__qir__u3(double 0x3FF921FB54B4C996, double 0.000000e+00, double 0.000000e+00, %Qubit* %1), !dbg !7
  call void @__quantum__qir__u3(double 0x3FF921FB54442D17, double 0x3FF0C152386E7788, double 0xC004F1A6C6595250, %Qubit* %1), !dbg !7
  call void @__quantum__qir__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FF0C152386E7788, %Qubit* %0), !dbg !7
  call void @__quantum__qir__cnot(%Qubit* %1, %Qubit* %0), !dbg !7
  call void @__quantum__qir__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFF921FB54442D18, %Qubit* %0), !dbg !7
  call void @__quantum__qir__cnot(%Qubit* %1, %Qubit* %0), !dbg !7
  call void @__quantum__qir__u3(double 0x3FF921FB54442D17, double 0xBFE0C15237AB6B1F, double 0x3FE0C15237AB6B20, %Qubit* %1), !dbg !7
  ret void, !dbg !7
}

declare void @__quantum__rt__reset(%Qubit*)

declare %Result* @__quantum__rt__measure(%Qubit*)

define void @main() !dbg !9 {
  br label %1, !dbg !10

1:                                                ; preds = %4, %0
  %2 = phi i64 [ 0, %0 ], [ %7, %4 ]
  %3 = icmp slt i64 %2, 2, !dbg !10
  br i1 %3, label %4, label %8, !dbg !10

4:                                                ; preds = %1
  %5 = call %Qubit* @__quantum__rt__qubit_allocate(), !dbg !10
  %6 = getelementptr %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 0), i64 %2, !dbg !10
  store %Qubit* %5, %Qubit** %6, align 8, !dbg !10
  %7 = add i64 %2, 1, !dbg !10
  br label %1, !dbg !10

8:                                                ; preds = %1
  %9 = load %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !13
  call void @__quantum__qir__h(%Qubit* %9), !dbg !13
  %10 = load %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !14
  %11 = load %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 1), align 8, !dbg !14
  call void @__quantum__qir__cnot(%Qubit* %10, %Qubit* %11), !dbg !14
  %12 = load %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !15
  %13 = call %Result* @__quantum__rt__measure(%Qubit* %12), !dbg !15
  %14 = call %Result* @__quantum__rt__result_get_one(), !dbg !15
  %15 = call i1 @__quantum__rt__result_equal(%Result* %13, %Result* %14), !dbg !15
  %16 = sext i1 %15 to i64, !dbg !15
  store i64 %16, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @a, i64 0, i64 0), align 4, !dbg !16
  %17 = load %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 1), align 8, !dbg !17
  %18 = call %Result* @__quantum__rt__measure(%Qubit* %17), !dbg !17
  %19 = call %Result* @__quantum__rt__result_get_one(), !dbg !17
  %20 = call i1 @__quantum__rt__result_equal(%Result* %18, %Result* %19), !dbg !17
  %21 = sext i1 %20 to i64, !dbg !17
  store i64 %21, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @a, i64 0, i64 1), align 4, !dbg !18
  %22 = load i64, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @a, i64 0, i64 0), align 4, !dbg !19
  %23 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), i64 %22), !dbg !20
  %24 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @nl, i64 0, i64 0)), !dbg !20
  %25 = load i64, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @a, i64 0, i64 1), align 4, !dbg !21
  %26 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), i64 %25), !dbg !22
  %27 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @nl, i64 0, i64 0)), !dbg !22
  br label %28, !dbg !10

28:                                               ; preds = %31, %8
  %29 = phi i64 [ 0, %8 ], [ %34, %31 ]
  %30 = icmp slt i64 %29, 2, !dbg !10
  br i1 %30, label %31, label %35, !dbg !10

31:                                               ; preds = %28
  %32 = getelementptr %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 0), i64 %29, !dbg !10
  %33 = load %Qubit*, %Qubit** %32, align 8, !dbg !10
  call void @__quantum__rt__qubit_release(%Qubit* %33), !dbg !10
  %34 = add i64 %29, 1, !dbg !10
  br label %28, !dbg !10

35:                                               ; preds = %28
  ret void, !dbg !23
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "__quantum__qir__rs", linkageName: "__quantum__qir__rs", scope: null, file: !4, type: !5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "../tests/out2.mlir", directory: "/Users/huazhelou/Documents/quantum/llvm/isqv2/mlir/build")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 0, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = distinct !DISubprogram(name: "main", linkageName: "main", scope: null, file: !4, line: 12, type: !5, scopeLine: 12, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!10 = !DILocation(line: 8, column: 6, scope: !11)
!11 = !DILexicalBlockFile(scope: !9, file: !12, discriminator: 0)
!12 = !DIFile(filename: "main.isq", directory: "/Users/huazhelou/Documents/quantum/llvm/isqv2/mlir/build")
!13 = !DILocation(line: 12, column: 9, scope: !11)
!14 = !DILocation(line: 13, column: 9, scope: !11)
!15 = !DILocation(line: 14, column: 16, scope: !11)
!16 = !DILocation(line: 14, column: 9, scope: !11)
!17 = !DILocation(line: 15, column: 16, scope: !11)
!18 = !DILocation(line: 15, column: 9, scope: !11)
!19 = !DILocation(line: 16, column: 15, scope: !11)
!20 = !DILocation(line: 16, column: 9, scope: !11)
!21 = !DILocation(line: 17, column: 15, scope: !11)
!22 = !DILocation(line: 17, column: 9, scope: !11)
!23 = !DILocation(line: 10, column: 1, scope: !11)

