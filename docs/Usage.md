Usage
=========================

We packed all tools into a Docker container.

```bash
alias isqv2="docker run -i --rm isqv2:<tag>"
alias isqc="isqv2 isqc"
alias isq-opt="isqv2 isq-opt"
alias mlir-translate="isqv2 mlir-translate"
alias simulator="isqv2 simulator"
```

isQ Compiler Frontend
-------------------------

```bash
isqc < test1.isq > test1.mlir
```

isQ MLIR Compiler
-------------------------

```bash
isq-opt -pass-pipeline="canonicalize,cse,isq-fold-constant-decorated-gates,isq-decompose-known-gates-qsd,isq-expand-decomposition,isq-lower-to-qir-rep,cse,canonicalize,isq-lower-qir-rep-to-llvm,canonicalize,cse,symbol-dce,llvm-legalize-for-export"  --mlir-print-debuginfo < test1.mlir > test1_llvm.mlir
mlir-translate --mlir-to-llvmir < test1_llvm.mlir > test1.ll
```

isQ Simulator
-------------------------

TODO: Building.

TODO: You need to mount a local volume into the container for accessing.