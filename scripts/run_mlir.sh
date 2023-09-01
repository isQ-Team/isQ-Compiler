#! /usr/bin/bash
# run.sh xxx.mlir
FILENAME="${1%%.*}"
EXTENSION="${1#*.}"
if [ "$EXTENSION" == "mlir" ]; then
    cat $1 | ${ISQ_ROOT}/bin/isq-opt --pass-pipeline=builtin.module\(cse,logic-lower-to-isq,isq-state-preparation,isq-oracle-decompose,isq-lower-switch,isq-recognize-famous-gates,isq-eliminate-neg-ctrl,isq-convert-famous-rot,canonicalize,cse,isq-pure-gate-detection,canonicalize,isq-fold-decorated-gates,canonicalize,isq-decompose-ctrl-u3,isq-convert-famous-rot,isq-decompose-known-gates-qsd,isq-remove-trivial-sq-gates,isq-expand-decomposition,canonicalize,cse\) --mlir-print-debuginfo > $FILENAME.opt.mlir
    if [ $? != 0 ]; then
        exit
    fi
    EXTENSION="opt.mlir"
fi
if [ "$EXTENSION" == "opt.mlir" ]; then
    cat $FILENAME.opt.mlir | ${ISQ_ROOT}/bin/isq-opt --pass-pipeline=builtin.module\(cse,isq-remove-gphase,lower-affine,isq-lower-to-qir-rep,cse,canonicalize,func.func\(convert-math-to-llvm\),arith-expand,expand-strided-metadata,memref-expand,convert-math-to-funcs,isq-lower-qir-rep-to-llvm,canonicalize,cse,symbol-dce,llvm-legalize-for-export,global-thread-local\) --mlir-print-debuginfo > $FILENAME.ll.mlir
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
    ${ISQ_ROOT}/bin/simulator --naive -e __isq__entry $FILENAME.so
fi
