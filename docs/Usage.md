Usage
=========================

We packed all tools into a tarball.

Preparation
-------------------------

Note that since our tarball depends on (nix-user-chroot)[https://github.com/nix-community/nix-user-chroot], you need to make sure that your kernel is configured with `CONFIG_USER_NS=y`.

To check: 

```bash
$ unshare --user --pid echo YES
YES
```

`run` is a thin wrapper around `nix-user-chroot`.

```bash
$ ./run
isQv2 Toolchain wrapper.
Usage: ./run [TOOL_NAME]
Tools directory: ./nix/store/8rsrr6dc3kl20g5gi7kclbv409m1kq7l-isqv2/bin/
# Lists all tools available.
$ ls ./nix/store/8rsrr6dc3kl20g5gi7kclbv409m1kq7l-isqv2/bin/
isqc isq-opt mlir-translate ...
```


We use these aliases:

```bash
alias isqv2="./run"
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