; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%Qubit = type opaque
%Result = type opaque

@frmt_spec = internal constant [4 x i8] c"%i \00"
@q = local_unnamed_addr global [2 x %Qubit*] undef
@a = local_unnamed_addr global [2 x i64] undef

; Function Attrs: nofree nounwind
declare noundef i32 @printf(i8* nocapture noundef readonly, ...) local_unnamed_addr #0

declare void @__quantum__rt__qubit_release(%Qubit*) local_unnamed_addr

declare %Qubit* @__quantum__rt__qubit_allocate() local_unnamed_addr

declare void @__quantum__qir__h(%Qubit*) local_unnamed_addr

declare void @__quantum__qir__cnot(%Qubit*, %Qubit*) local_unnamed_addr

declare void @__quantum__qir__u3(double, double, double, %Qubit*) local_unnamed_addr

declare void @__quantum__qir__gphase(double) local_unnamed_addr

define void @__quantum__qir__rs(%Qubit* %0, %Qubit* %1) local_unnamed_addr !dbg !3 {
  tail call void @__quantum__qir__gphase(double 0x3FE0C15237AB6B20), !dbg !7
  tail call void @__quantum__qir__u3(double 0x3FF921FB54B4C996, double 0xBFF921FB54442D18, double 0x400921FB54442D18, %Qubit* %1), !dbg !7
  tail call void @__quantum__qir__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFF921FB54442D18, %Qubit* %0), !dbg !7
  tail call void @__quantum__qir__cnot(%Qubit* %1, %Qubit* %0), !dbg !7
  tail call void @__quantum__qir__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FF921FB54442D18, %Qubit* %0), !dbg !7
  tail call void @__quantum__qir__cnot(%Qubit* %1, %Qubit* %0), !dbg !7
  tail call void @__quantum__qir__u3(double 0x3FF921FB54B4C996, double 0.000000e+00, double 0.000000e+00, %Qubit* %1), !dbg !7
  tail call void @__quantum__qir__u3(double 0x3FF921FB54442D17, double 0x3FF0C152386E7788, double 0xC004F1A6C6595250, %Qubit* %1), !dbg !7
  tail call void @__quantum__qir__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FF0C152386E7788, %Qubit* %0), !dbg !7
  tail call void @__quantum__qir__cnot(%Qubit* %1, %Qubit* %0), !dbg !7
  tail call void @__quantum__qir__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFF921FB54442D18, %Qubit* %0), !dbg !7
  tail call void @__quantum__qir__cnot(%Qubit* %1, %Qubit* %0), !dbg !7
  tail call void @__quantum__qir__u3(double 0x3FF921FB54442D17, double 0xBFE0C15237AB6B1F, double 0x3FE0C15237AB6B20, %Qubit* %1), !dbg !7
  ret void, !dbg !7
}

declare %Result* @__quantum__rt__measure(%Qubit*) local_unnamed_addr

define void @main() local_unnamed_addr !dbg !9 {
  %1 = tail call %Qubit* @__quantum__rt__qubit_allocate(), !dbg !10
  store %Qubit* %1, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !10
  %2 = tail call %Qubit* @__quantum__rt__qubit_allocate(), !dbg !10
  store %Qubit* %2, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 1), align 8, !dbg !10
  %3 = load %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !13
  tail call void @__quantum__qir__h(%Qubit* %3), !dbg !13
  %4 = load %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !14
  %5 = load %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 1), align 8, !dbg !14
  tail call void @__quantum__qir__cnot(%Qubit* %4, %Qubit* %5), !dbg !14
  %6 = load %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !15
  %7 = tail call %Result* @__quantum__rt__measure(%Qubit* %6), !dbg !15
  %8 = bitcast %Result* %7 to i1*, !dbg !15
  %9 = load i1, i1* %8, align 1, !dbg !15
  %10 = sext i1 %9 to i64, !dbg !15
  store i64 %10, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @a, i64 0, i64 0), align 8, !dbg !16
  %11 = load %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 1), align 8, !dbg !17
  %12 = tail call %Result* @__quantum__rt__measure(%Qubit* %11), !dbg !17
  %13 = bitcast %Result* %12 to i1*, !dbg !17
  %14 = load i1, i1* %13, align 1, !dbg !17
  %15 = sext i1 %14 to i64, !dbg !17
  store i64 %15, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @a, i64 0, i64 1), align 8, !dbg !18
  %16 = load i64, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @a, i64 0, i64 0), align 8, !dbg !19
  %17 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), i64 %16), !dbg !20
  %putchar = tail call i32 @putchar(i32 10), !dbg !20
  %18 = load i64, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @a, i64 0, i64 1), align 8, !dbg !21
  %19 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), i64 %18), !dbg !22
  %putchar2 = tail call i32 @putchar(i32 10), !dbg !22
  %20 = load %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !10
  tail call void @__quantum__rt__qubit_release(%Qubit* %20), !dbg !10
  %21 = load %Qubit*, %Qubit** getelementptr inbounds ([2 x %Qubit*], [2 x %Qubit*]* @q, i64 0, i64 1), align 8, !dbg !10
  tail call void @__quantum__rt__qubit_release(%Qubit* %21), !dbg !10
  ret void, !dbg !23
}

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #0

attributes #0 = { nofree nounwind }

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

