Usage
=========================

We packed all tools into a tarball.

Preparation
-------------------------


### Requirements

Note that since our tarball depends on (nix-user-chroot)[https://github.com/nix-community/nix-user-chroot], you need to make sure that your kernel is configured with `CONFIG_USER_NS=y`.

To check: 

```bash
$ unshare --user --pid echo YES
YES
```

### Using with Docker containers.

There are two approaches for using toolchain with Docker containers.

The first approach is to allow `unshare` in a container by granting `SYS_ADMIN` capability:

```bash
$ docker run --rm --cap-add SYS_ADMIN -v `pwd`:/isq -it ubuntu bash -c 'cd /isq && ./run isqc -v'
isqc (isQ Compiler)
```

The second approach is simpler: just throw the `nix` folder into root folder as `/nix`, and the commands under the folder followed by "`Tools directory:`" when running `run` (see Command Usage) will be available. In this case, the `run` wrapper, as well as `SYS_ADMIN` capability, is no longer required.

```bash
$ docker run --rm -v `pwd`/nix:/nix -it ubuntu /nix/store/8rsrr6dc3kl20g5gi7kclbv409m1kq7l-isqv2/bin/isqc -v
isqc (isQ Compiler)
```


### Command usage

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