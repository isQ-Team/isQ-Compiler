

(Deprecated) Step-by-step usages
========================

The steps below are for advances usages, e.g. testing each step.


isQ Compiler Frontend
-------------------------

```bash
isqc1 < test1.isq > test1.mlir
```

isQ MLIR Compiler
-------------------------

```bash
isq-opt -pass-pipeline="canonicalize,cse,isq-fold-constant-decorated-gates,isq-decompose-known-gates-qsd,isq-expand-decomposition,isq-lower-to-qir-rep,cse,canonicalize,isq-lower-qir-rep-to-llvm,canonicalize,cse,symbol-dce,llvm-legalize-for-export"  --mlir-print-debuginfo < test1.mlir > test1_llvm.mlir
mlir-translate --mlir-to-llvmir < test1_llvm.mlir > test1.ll
```

isQ Simulator
-------------------------

```bash
cat > test1.c <<EOF
void __isq__entry();
void isq_simulator_entry(){
  __isq__entry();
}
EOF
clang test1.c test1.ll $ISQ_ROOT/share/isq-simulator/isq-simulator.bc -O3 -shared -fPIC -o test1.so
RUST_LOG=INFO simulator ./test1.so
```