#! /usr/bin/bash
# compile.sh xxx.mlir
FILENAME="${1%%.*}"
EXTENSION="${1#*.}"
if [ "$EXTENSION" == "mlir" ]; then
    ${ISQ_ROOT}/bin/isq-opt --pass-pipeline=isq-recognize-famous-gates,isq-eliminate-neg-ctrl,isq-convert-famous-rot,canonicalize,cse,isq-pure-gate-detection,canonicalize,isq-fold-decorated-gates,canonicalize,isq-decompose-ctrl-u3,isq-convert-famous-rot,isq-decompose-known-gates-qsd,isq-remove-trivial-sq-gates,isq-expand-decomposition,canonicalize,cse --mlir-print-debuginfo $1 -o $FILENAME.opt.mlir
    if [ $? != 0 ]; then
        exit
    fi
    EXTENSION="opt.mlir"
fi
if [ "$EXTENSION" == "opt.mlir" ]; then
    ${ISQ_ROOT}/bin/isq-opt --pass-pipeline=cse,isq-remove-gphase,lower-affine,isq-lower-to-qir-rep,cse,canonicalize,builtin.func\(convert-math-to-llvm\),isq-lower-qir-rep-to-llvm,canonicalize,cse,symbol-dce,llvm-legalize-for-export --mlir-print-debuginfo $FILENAME.opt.mlir -o $FILENAME.ll.mlir
    if [ $? != 0 ]; then
        exit
    fi
    EXTENSION="ll.mlir"
fi
if [ "$EXTENSION" == "ll.mlir" ]; then
    mlir-translate --mlir-to-llvmir $FILENAME.ll.mlir -o $FILENAME.ll
    EXTENSION="ll"
fi
if [ "$EXTENSION" == "ll" ]; then
    llvm-link $FILENAME.ll ${ISQ_ROOT}/share/isq-simulator/isq-simulator.bc -o $FILENAME.link
    opt $FILENAME.link -o $FILENAME.opt
    llc --filetype=obj $FILENAME.opt -o $FILENAME.o
    lld -flavor gnu -shared $FILENAME.o -o $FILENAME.so
fi
